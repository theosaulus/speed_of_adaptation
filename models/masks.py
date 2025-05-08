import torch
from models.cpdag_utils import compute_cpdag

def fully_connected_mask(num_nodes):
    mask = torch.ones(num_nodes, num_nodes)
    mask.fill_diagonal_(0)
    return mask

def causal_mask(true_graph):
    # true_graph is a torch.Tensor or numpy array adjacency
    return true_graph.clone().detach().to(torch.float32)

def anti_causal_mask(true_graph):
    return causal_mask(true_graph).transpose(0, 1)

def skeleton_mask(true_graph):
    A = causal_mask(true_graph)
    return ((A + A.transpose(0,1)) > 0).float()

def cpdag_mask(true_graph):
    # compute_cpdag should return a numpy array or torch.Tensor
    cpdag = compute_cpdag(true_graph)
    return torch.tensor(cpdag, dtype=torch.float32)

def create_mask(graph, mask='causal', device=None):
    """
    Build the [N,N] mask from a CausalDAG `graph` and a mask type.

    mask ∈ {'fully_connected', 'causal', 'anti_causal', 'skeleton', 'cpdag'}
    """
    # get adjacency as a torch.Tensor
    A = torch.tensor(graph.adj_matrix, dtype=torch.float32, device=device)

    if mask == 'fully_connected':
        return fully_connected_mask(A.size(0))
    elif mask == 'causal':
        return causal_mask(A)
    elif mask == 'anti_causal':
        return anti_causal_mask(A)
    elif mask == 'skeleton':
        return skeleton_mask(A)
    elif mask == 'cpdag':
        return cpdag_mask(A)
    else:
        raise ValueError(f"Unknown mask {mask!r}. "
                         "Expected one of "
                         "['fully_connected','causal','anti_causal','skeleton','cpdag'].")
