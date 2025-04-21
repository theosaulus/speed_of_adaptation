import numpy as np
import torch
import torch.nn.functional as F

from objectives.pseudo_ll import pseudo_ll_loss
from data.dataset_creation import sample_dict_to_tensor

def compute_nll_ground_truth(graph, X):
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

def evaluate_zero_shot(model, graph, dataset, order, device):
    """
    Zero-shot evaluation across all regimes (observational + each intervention).
    Returns a dict with per‑regime NLL and bounds and their means.
    """
    results = {}
    
    # Observational
    X_obs, _ = sample_dict_to_tensor(dataset['observational'], order)
    X_obs = X_obs.to(device)
    results['nll_obs']   = pseudo_ll_loss(model, X_obs).item()
    results['bound_obs'] = compute_nll_ground_truth(graph, X_obs)

    # Interventional (one task per intervened variable)
    nlls = []
    bounds = []
    for var_name, sample_dict in dataset['interventional'].items():
        X_int, _ = sample_dict_to_tensor(sample_dict, order)
        X_int = X_int.to(device)
        nll = pseudo_ll_loss(model, X_int).item()
        bnd = compute_nll_ground_truth(graph, X_int)
        results[f'nll_{var_name}']   = nll
        results[f'bound_{var_name}'] = bnd
        nlls.append(nll)
        bounds.append(bnd)

    # Means
    all_nlls   = [results['nll_obs']] + nlls
    all_bounds = [results['bound_obs']] + bounds
    results['nll_mean']   = float(np.mean(all_nlls))
    results['bound_mean'] = float(np.mean(all_bounds))

    return results
