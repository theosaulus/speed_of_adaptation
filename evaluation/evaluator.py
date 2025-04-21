import numpy as np
import torch
import torch.nn.functional as F

from objectives.pseudo_ll import pseudo_ll_loss
from data.dataset_creation import sample_dict_to_tensor

def compute_nll_bound(graph, X):
    """
    Compute the NLL-Mean under the ground-truth SCM.
    X: torch.Tensor of shape [batch_size, num_vars], in same variable order as graph.variables
    """
    X_np = X.detach().cpu().numpy()
    nlls = []
    for sample in X_np:
        nll = 0.0
        for i, var in enumerate(graph.variables):
            parents = np.where(graph.adj_matrix[:, i])[0]
            parent_vals = {graph.variables[p].name: int(sample[p]) for p in parents}
            xi = int(sample[i])
            prob = var.prob_dist.prob(parent_vals, xi)
            nll += -np.log(prob + 1e-12)
        nlls.append(nll / len(graph.variables))
    return float(np.mean(nlls))

def compute_nll_on_ground_truth(model, graph, X):
    """
    Compute the NLL-Mean of the model predictions under the ground-truth SCM.
    """
    outputs = model(X)
    outputs_np = outputs.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    nlls = []
    for idx, sample in enumerate(X_np):
        nll = 0.0
        for i, var in enumerate(graph.variables):
            parents = np.where(graph.adj_matrix[:, i])[0]
            parent_vals = {graph.variables[p].name: int(sample[p]) for p in parents}
            xi = int(outputs_np[idx, i])
            prob = var.prob_dist.prob(parent_vals, xi)
            nll += -np.log(prob + 1e-12)
        nlls.append(nll / len(graph.variables))
    return float(np.mean(nlls))

def evaluate_zero_shot(model, graph, dataset, order, device):
    """
    Zero-shot evaluation across all regimes (observational + each intervention).
    Returns a dict with per‑regime NLL and bounds and their means.
    """
    results = {}

    # Observational
    X_obs, _ = sample_dict_to_tensor(dataset['observational'], order)
    X_obs = X_obs.to(device)
    results['raw_pseudo_nll'] = pseudo_ll_loss(model, X_obs).item()
    results['bound_obs'] = compute_nll_bound(graph, X_obs)

    # Interventional (one task per intervened variable)
    raw_nlls = []
    nlls_on_gt = []
    bounds = []
    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, order)
        X_int = X_int.to(device)

        raw_nll = pseudo_ll_loss(model, X_int).item()
        nll_on_gt = compute_nll_on_ground_truth(model, graph, X_int)
        bound = compute_nll_bound(graph, X_int)

        results[f'raw_nll_{var_name}'] = raw_nll
        results[f'nll_on_gt_{var_name}'] = nll_on_gt
        results[f'bound_{var_name}'] = bound

        raw_nlls.append(raw_nll)
        nlls_on_gt.append(nll_on_gt)
        bounds.append(bound)

    # Means
    all_raw_nlls = [results['raw_pseudo_nll']] + raw_nlls
    all_nlls_on_gt = [results['raw_pseudo_nll']] + nlls_on_gt
    all_bounds = [results['bound_obs']] + bounds

    results['raw_nll_mean'] = float(np.mean(all_raw_nlls))
    results['nll_on_gt_mean'] = float(np.mean(all_nlls_on_gt))
    results['bound_mean'] = float(np.mean(all_bounds))

    return results
