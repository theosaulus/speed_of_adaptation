import numpy as np
import torch
import torch.nn.functional as F

def compute_nll_bound(graph_src, graph_tgt, X, order, var_indices=None):
    """
    Compute the NLL-Mean under the ground-truth SCM.
    X and order have the same ordering
    graph and var_indices have the same ordering
    They comunicate with order.index(var)

    graph_tgt: the ground truth graph to compute the NLL on
    X: torch.Tensor of shape [batch_size, num_vars], in same variable order as order, not graph.variables
    var_indices: optional iterable of variable indices to include; if None, uses all variables.
    """
    X_np = X.detach().cpu().numpy()
    B, N = X_np.shape

    # default = all variables in the graph
    if var_indices is None:
        var_indices = list(range(len(graph_tgt.variables)))

    nll_per_sample = np.zeros(B, dtype=np.float64)

    for i in var_indices:
        var = graph_tgt.variables[i]
        col = order.index(var.name) # which column of X_np is this var?
        
        # p_tgt = CPD under the intervened (true) graph
        pa_tgt = np.where(graph_tgt.adj_matrix[:, i] != 0)[0]
        inputs_tgt = {
            graph_tgt.variables[p].name: X_np[:, order.index(graph_tgt.variables[p].name)]
            for p in pa_tgt
        }
        p_tgt = var.prob_dist.prob_func(inputs_tgt, B)
        if p_tgt.ndim == 1:
            p_tgt = np.tile(p_tgt[None, :], (B, 1))

        # q_src = CPD under the observational graph
        pa_src = np.where(graph_src.adj_matrix[:, i] != 0)[0]
        inputs_src = {
            graph_src.variables[p].name: X_np[:, order.index(graph_src.variables[p].name)]
            for p in pa_src
        }
        q_src = graph_src.variables[i].prob_dist.prob_func(inputs_src, B)
        if q_src.ndim == 1:
            q_src = np.tile(q_src[None, :], (B, 1))

        # accumulate cross‐entropy: -sum p_tgt log q_src
        nll_per_sample += -np.sum(p_tgt * np.log(q_src + 1e-12), axis=1)

    # average over variables and samples
    nll_per_sample /= float(len(var_indices))
    return float(nll_per_sample.mean())


def compute_nll_on_ground_truth(model, graph, X, order, params=None, var_indices=None):
    """
    Compute the NLL-Mean of the model predictions under the ground-truth SCM.
    """
    probs = model(X, params=params) # [batch_size, num_vars, output_dim]
    batch_size, num_vars, output_dim = probs.shape

    if var_indices is None:
        var_indices = list(range(len(graph.variables)))

    p_model = probs.detach().cpu().numpy() # (B, N, K)
    X_np = X.detach().cpu().numpy() # (B, N)
    nll_per_sample = np.zeros(batch_size, dtype=np.float64)

    for i in var_indices:
        var = graph.variables[i]
        col = order.index(var.name)

        # q_model_i = preditcion of the model
        q_model_i = p_model[:, col, :] # (B, K)

        parents = np.where(graph.adj_matrix[:, i] != 0)[0]
        parents_inputs = {
            graph.variables[p].name: X_np[:, order.index(graph.variables[p].name)]
            for p in parents
        }
        p_groundtruth = var.prob_dist.prob_func(parents_inputs, batch_size)
        if p_groundtruth.shape[0] != batch_size: # in case there is no parents, expand to batch size
            p_groundtruth = np.tile(p_groundtruth, (batch_size, 1)) # (B, num_categs)

        nll_per_sample += -np.sum(p_groundtruth * np.log(q_model_i + 1e-12), axis=1)
    
    nll_per_sample /= float(len(var_indices))
    return float(nll_per_sample.mean())
