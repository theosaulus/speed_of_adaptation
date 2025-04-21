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
    logits = model(X, params=params)
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    X_np = X.detach().cpu().numpy()
    nlls = []
    for idx, sample in enumerate(X_np):
        nll = 0.0
        for i, var in enumerate(graph.variables):
            parents = np.where(graph.adj_matrix[:, i])[0]
            parent_vals = {graph.variables[p].name: int(sample[p]) for p in parents}
            xi = int(preds[idx, i])
            prob = var.prob_dist.prob(parent_vals, xi)
            nll += -np.log(prob + 1e-12)
        nlls.append(nll / len(graph.variables))
    return float(np.mean(nlls))
