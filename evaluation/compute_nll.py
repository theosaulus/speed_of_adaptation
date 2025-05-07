import numpy as np
import torch
import torch.nn.functional as F

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
    batch_size, num_vars, output_dim = probs.shape
    
    p_model = probs.detach().cpu().numpy() # (B, N, K)
    X_np = X.detach().cpu().numpy() # (B, N)
    nll_per_sample = np.zeros(batch_size, dtype=np.float64)

    for i, var in enumerate(graph.variables):
        p_model_i = p_model[:, i, :] # (B, K)

        parents = np.where(graph.adj_matrix[:, i])[0]
        parents_inputs = {
            graph.variables[p].name: X_np[:, p]
            for p in parents
        }
        p_groundtruth = var.prob_dist.prob_func(parents_inputs, batch_size)
        
        if p_model_i.shape[-1] != p_groundtruth.shape[-1]:
            p_groundtruth = F.one_hot(torch.tensor(p_groundtruth, dtype=torch.long), num_classes=output_dim).cpu().numpy() # (B, K)
        nll_per_sample += -np.sum(p_model_i * np.log(p_groundtruth + 1e-12), axis=1)
    
    nll_per_sample /= num_vars
    return float(nll_per_sample.mean())
