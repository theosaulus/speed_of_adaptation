import numpy as np
import torch

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

def compute_nll_on_ground_truth(model, graph, X, params=None):
    """
    Compute the NLL-Mean of the model predictions under the ground-truth SCM.
    """
    probs = model(X, params=params) # [batch_size, num_vars, output_dim]
    output_dim = probs.shape[-1] if len(probs.shape) == 3 else 1
    X_np = X.detach().cpu().numpy()
    nlls = []
    for idx, sample in enumerate(X_np):
        nll = 0.0
        for i, var in enumerate(graph.variables):
            for k in range(output_dim):
                p_model = probs[idx, i, k]
                parents = np.where(graph.adj_matrix[:, i])[0]
                parent_vals = {graph.variables[p].name: int(sample[p]) for p in parents}
                p_groundtruth = var.prob_dist.prob(parent_vals, k)
                nll += -p_model * np.log(p_groundtruth + 1e-12)
            nlls.append(nll / len(graph.variables))
    return float(np.mean(nlls))
