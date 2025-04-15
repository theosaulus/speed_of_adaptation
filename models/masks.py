import torch
from models.cpdag_utils import compute_cpdag

def fully_connected_mask(num_nodes):
    mask = torch.ones(num_nodes)
    mask.fill_diagonal_(0) # remove self-loops
    return mask

def causal_mask(true_graph):
    return true_graph

def anti_causal_mask(true_graph):
    return true_graph.transpose(0, 1)

def skeleton_mask(true_graph):
    return ((true_graph + true_graph.transpose(0, 1)) > 0).float()

def cpdag_mask(true_graph):
    cpdag = compute_cpdag(true_graph)
    return cpdag
