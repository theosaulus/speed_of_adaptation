import numpy as np
import torch
import torch.nn.functional as F

from objectives.pseudo_ll import pseudo_ll_loss
from data.dataset_creation import sample_dict_to_tensor
from evaluation.compute_nll import compute_nll_bound, compute_nll_on_ground_truth

def evaluate_bounds(graph, dataset, order, device):
    results = {}
    X_obs, _ = sample_dict_to_tensor(dataset['observational'], order)
    X_obs = X_obs.to(device)
    results['bound_obs'] = compute_nll_bound(graph, X_obs)

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
    Returns a dict with per‑regime NLL and bounds and their means.
    """
    results = {}
    X_obs, _ = sample_dict_to_tensor(dataset['observational'], order)
    X_obs = X_obs.to(device)
    results['raw_pseudo_nll_obs'] = - pseudo_ll_loss(model, X_obs).item()
    results['nll_on_gt_obs'] = compute_nll_on_ground_truth(model, graph, X_obs).item()

    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, order)
        X_int = X_int.to(device)

        raw_nll = - pseudo_ll_loss(model, X_int).item()
        nll_on_gt = compute_nll_on_ground_truth(model, graph, X_int)

        results[f'raw_pseudo_nll_int_{var_name}'] = raw_nll
        results[f'nll_on_gt_int_{var_name}'] = nll_on_gt

    return results

def evaluate_few_shot(
        model, 
        graph, 
        dataset, 
        order, 
        few_shot_num_samples, 
        few_shot_gradient_steps, 
        device
    ):
    """
    Few-shot evaluation across all regimes (observational + each intervention).
    Returns a dict with per‑regime NLL and bounds and their means, for the list 
    of shots and the given number of gradient steps.
    """
    results = {}

    regimes = {'obs': dataset['observational']}
    regimes.update(dataset['interventional'])

    for K in few_shot_num_samples:
        for name, sample_dict in regimes.items():
            X, _ = sample_dict_to_tensor(sample_dict, order)
            X = X.to(device)
            N = X.size(0)

            # if not enough samples error
            if K >= N:
                raise ValueError(f"Not enough samples for {name} regime: {N} < {K}")
            
            else:
                # pick K samples for adaptation
                idx = np.random.choice(N, K, replace=False)
                X_inner = X[idx]
                # params = OrderedDict(model.meta_named_parameters())
                # for _ in range(few_shot_gradient_steps):
                #     loss_i = - pseudo_ll_loss(model, X_inner, params=params)
                #     params = gradient_update_parameters(
                #         model,
                #         loss_i,
                #         params=params,
                #         step_size=config['objective']['inner_lr'],
                #         first_order=config['objective']['maml_first_order']
                #     )
                # adapted_params = params

            raw_nll = - pseudo_ll_loss(model, X, params=adapted_params).item()
            nll_on_gt = compute_nll_on_ground_truth(model, graph, X, params=adapted_params)

            results[f'raw_nll_{name}_shot{K}'] = raw_nll
            results[f'nll_on_gt_{name}_shot{K}'] = nll_on_gt

    return results