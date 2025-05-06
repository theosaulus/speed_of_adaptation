import numpy as np
import torch

def sample_dict_to_tensor(sample, device, order=None):
    """
    Convert a sample (dictionary) from graph.sample() into a tensor.
    
    Parameters:
        sample: dict
            A dictionary with keys (variable names) and values (arrays) as returned by graph.sample().
        order: list or None
            The canonical order of variable names. If None, keys are sorted alphabetically.
            
    Returns:
        tensor: torch.LongTensor of shape (batch_size, num_vars)
    """
    if order is None:
        order = sorted(sample.keys())
    sample_list = []
    for key in order:
        value = np.array(sample[key]).reshape(-1)
        sample_list.append(value)
    data = np.stack(sample_list, axis=1)
    return torch.tensor(data, dtype=torch.long, device=device), order

def build_dataset(graph, num_obs, num_int):
    """
    Build a dataset for a given graph with observational data and interventional data.

    For observational data, samples are generated with no interventions.
    For interventional data, for every variable in the graph a set of samples is generated
    by intervening on that variable (by assigning randomly drawn values).

    Parameters:
        graph: an instance of CausalDAG
            The causal graph with ground-truth CPDs.
        num_obs: int
            Number of observational samples to generate.
        num_int: int
            Number of interventional samples per variable.

    Returns:
        dataset: dict, with keys:
            "observational": the observational sample (a dictionary as returned by graph.sample())
            "interventional": a dict mapping variable names to interventional samples (each as a dict)
    """
    # Generate observational data
    obs_sample = graph.sample(batch_size=num_obs)
    
    # Generate interventional data for each variable
    interventional = {}
    for var in graph.variables:
        try:
            n_categs = var.prob_dist.num_categs
        except AttributeError:
            n_categs = int(obs_sample[var.name].max()) + 1 # assuming categorical variables are indexed 0, 1, ..., n_categs-1
        intervention_values = np.random.randint(0, n_categs, size=(num_int,))
        int_sample = graph.sample(interventions={var.name: intervention_values},
                                  batch_size=num_int)
        interventional[var.name] = int_sample

    dataset = {
        "observational": obs_sample,
        "interventional": interventional
    }
    return dataset
