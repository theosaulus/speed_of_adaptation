import numpy as np
import torch
import random
import os

def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_dataset_graph_pairs(folder_path):
    files = os.listdir(folder_path)

    datasets = {}
    graphs = {}

    for f in files:
        if f.startswith("dataset_") and f.endswith("_random.pt"):
            idx = f[len("dataset_"):f.rfind("_random.pt")]
            datasets[idx] = os.path.join(folder_path, f)
        elif f.startswith("graph_") and f.endswith("_random.pt"):
            idx = f[len("graph_"):f.rfind("_random.pt")]
            graphs[idx] = os.path.join(folder_path, f)

    # Find common indices
    common_indices = sorted(set(datasets.keys()) & set(graphs.keys()))

    for idx in common_indices:
        yield datasets[idx], graphs[idx]