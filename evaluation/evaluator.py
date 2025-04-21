import numpy as np
import copy
import torch
import torch.nn.functional as F

from objectives.pseudo_ll import pseudo_ll_loss
from data.dataset_creation import sample_dict_to_tensor
from evaluation.compute_nll import compute_nll_bound, compute_nll_on_ground_truth

def evaluate_bounds(graph, dataset, order, device):
    results = {}
    
    # Observational regime
    X_obs, _ = sample_dict_to_tensor(dataset['observational'], order)
    X_obs = X_obs.to(device)
    results['bound_obs'] = compute_nll_bound(graph, X_obs)

    # Interventional regimes
    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, order)
        X_int = X_int.to(device)

        graph_int = graph.copy()
        graph_int = graph_int.get_intervened_graph({var_name: X_int[:, graph.variables.index(var_name)]})
        
        bound_int = compute_nll_bound(graph_int, X_int)
        results[f'bound_{var_name}'] = bound_int

    return results    

def evaluate_zero_shot(model, graph, dataset, order, device):
    """
    Zero-shot evaluation across all regimes (observational + each intervention).
    Returns a dict with per‑regime NLL.
    """
    results = {}

    # Observational regime
    X_obs, _ = sample_dict_to_tensor(dataset['observational'], order)
    X_obs = X_obs.to(device)
    results['raw_pseudo_nll_obs'] = - pseudo_ll_loss(model, X_obs).item()
    results['nll_on_gt_obs'] = compute_nll_on_ground_truth(model, graph, X_obs).item()

    # Interventional regimes
    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, order)
        X_int = X_int.to(device)

        graph_int = graph.copy()
        graph_int = graph_int.get_intervened_graph({var_name: X_int[:, graph.variables.index(var_name)]})

        with torch.no_grad():
            raw_nll = - pseudo_ll_loss(model, X_int).item()
            nll_on_gt_obs = compute_nll_on_ground_truth(model, graph, X_int)
            nll_on_gt_int = compute_nll_on_ground_truth(model, graph_int, X_int)

        results[f'raw_pseudo_nll_{var_name}'] = raw_nll
        results[f'nll_on_gt_obs_{var_name}'] = nll_on_gt_obs
        results[f'nll_on_gt_int_{var_name}'] = nll_on_gt_int

    return results

def evaluate_few_shot(
        model, 
        graph, 
        dataset, 
        order, 
        few_shot_num_samples, 
        few_shot_gradient_steps, 
        device,
        few_shot_lr=0.1,
    ):
    """
    Few-shot evaluation across all interventional regimes.
    Returns a dict with per‑regime NLL, for the list of shots and the given 
    number of gradient steps.
    """
    results = {}

    for K in few_shot_num_samples:
        for var_name, sample_dict in dataset['interventional'].items():
            X_int, _ = sample_dict_to_tensor(sample_dict, order)
            X_int = X_int.to(device)
            N = X_int.size(0)

            graph_int = graph.copy()
            graph_int = graph_int.get_intervened_graph({var_name: X_int[:, graph.variables.index(var_name)]})

            # if not enough samples error
            if K >= N:
                raise ValueError(f"Not enough samples for {var_name} regime: {N} < {K}")
            
            else:
                finetuned_model = copy.deepcopy(model).to(device).train()
                optimizer = torch.optim.SGD(finetuned_model.parameters(), lr=few_shot_lr)
                idx = np.random.choice(N, K, replace=False)
                X_adapt = X_int[idx]

                for _ in range(few_shot_gradient_steps):
                    loss_i = pseudo_ll_loss(finetuned_model, X_adapt)
                    optimizer.zero_grad()
                    loss_i.backward()
                    optimizer.step()
                
                finetuned_model.eval()

            with torch.no_grad():
                raw_nll = - pseudo_ll_loss(finetuned_model, X_int).item()
                nll_on_gt_obs = compute_nll_on_ground_truth(finetuned_model, graph, X_int)
                nll_on_gt_int = compute_nll_on_ground_truth(finetuned_model, graph_int, X_int)

            results[f'raw_pseudo_nll_{var_name}_{few_shot_gradient_steps}_shot_{K}_examples'] = raw_nll
            results[f'nll_on_gt_obs_{var_name}_{few_shot_gradient_steps}_shot_{K}_examples'] = nll_on_gt_obs
            results[f'nll_on_gt_int_{var_name}_{few_shot_gradient_steps}_shot_{K}_examples'] = nll_on_gt_int

    return results