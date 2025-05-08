import numpy as np
import torch
import torch.nn.functional as F

def compute_nll_bound(graph, X, order, var_indices=None):
    """
    Compute the NLL-Mean under the ground-truth SCM.
    X and order have the same ordering
    graph and var_indices have the same ordering
    They comunicate with order.index(var)

    X: torch.Tensor of shape [batch_size, num_vars], in same variable order as order, not graph.variables
    var_indices: optional iterable of variable indices to include; if None, uses all variables.
    """
    X_np = X.detach().cpu().numpy()
    batch_size, num_vars = X_np.shape

    if var_indices is None:
        var_indices = range(num_vars)
    var_indices = [order.index(graph.variables[i].name) for i in var_indices]

    nll_per_sample = np.zeros(batch_size, dtype=np.float64)

    for i, var in enumerate(graph.variables):
        var_idx = order.index(var.name)
        if var_idx not in var_indices:
            continue

        parents = np.where(graph.adj_matrix[:, i])[0]
        parents_inputs = {
            graph.variables[p].name: X_np[:, order.index(graph.variables[p].name)]
            for p in parents
        }

        p_groundtruth = var.prob_dist.prob_func(parents_inputs, batch_size) # (B, num_categs)
        if p_groundtruth.shape[0] != batch_size:
            # in case there is no parents, expand to batch size
            p_groundtruth = np.tile(p_groundtruth, (batch_size, 1)) # (B, num_categs)
        nll_per_sample += -np.sum(p_groundtruth * np.log(p_groundtruth + 1e-12), axis=1)
    
    nll_per_sample /= float(len(list(var_indices)))
    return float(nll_per_sample.mean())

    # nlls = []
    # for sample in X_np:
    #     nll = 0.0
    
    #     for i, var in enumerate(graph.variables):
    #         if i not in var_indices:
    #             continue
    #         parents = np.where(graph.adj_matrix[:, i])[0]
    #         parent_vals = {graph.variables[p].name: int(sample[p]) for p in parents}

    #         xi = int(sample[i])
    #         prob = var.prob_dist.prob(parent_vals, xi)
    #         nll += -np.log(prob + 1e-12)
    
    #     nlls.append(nll / float(len(list(var_indices))))
    # return float(np.mean(nlls))

def compute_nll_on_ground_truth(model, graph, X, order, params=None, var_indices=None):
    """
    Compute the NLL-Mean of the model predictions under the ground-truth SCM.
    """
    probs = model(X, params=params) # [batch_size, num_vars, output_dim]
    batch_size, num_vars, output_dim = probs.shape

    if var_indices is None:
        var_indices = range(num_vars)
    var_indices = [order.index(graph.variables[i].name) for i in var_indices]

    p_model = probs.detach().cpu().numpy() # (B, N, K)
    X_np = X.detach().cpu().numpy() # (B, N)
    nll_per_sample = np.zeros(batch_size, dtype=np.float64)

    for i, var in enumerate(graph.variables):
        var_idx = order.index(var.name)
        if var_idx not in var_indices:
            continue
        p_model_i = p_model[:, var_idx, :] # (B, K)

        parents = np.where(graph.adj_matrix[:, i])[0]
        parents_inputs = {
            graph.variables[p].name: X_np[:, order.index(graph.variables[p].name)]
            for p in parents
        }
        p_groundtruth = var.prob_dist.prob_func(parents_inputs, batch_size)
        if p_groundtruth.shape[0] != batch_size:
            # in case there is no parents, expand to batch size
            p_groundtruth = np.tile(p_groundtruth, (batch_size, 1)) # (B, num_categs)

        # if p_model_i.shape[-1] != p_groundtruth.shape[-1]:
        #     # in case of a constant distribution...
        #     p_groundtruth = F.one_hot(p_groundtruth.clone().detach().long(), num_classes=output_dim).cpu().numpy() # (B, K)
        # nll_per_sample += -np.sum(p_model_i * np.log(p_groundtruth + 1e-12), axis=1)
        nll_per_sample += -np.sum(p_groundtruth * np.log(p_model_i + 1e-12), axis=1)
    
    nll_per_sample /= float(len(list(var_indices)))
    return float(nll_per_sample.mean())
