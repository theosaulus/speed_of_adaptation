import numpy as np
import torch
from torch.optim import Adam

from data.dataset_creation import build_dataset, sample_dict_to_tensor
from data.graph_generation import generate_categorical_graph, get_graph_func
from models.model_factory import create_model
from models.masks import create_mask
from objectives.pseudo_ll import pseudo_ll_loss
from objectives.maml import maml_meta_update


def tasks_from_dataset(dataset, batch_size, order):
    """
    Turn observational + interventional data into a list of tasks,
    each task being a dict with 'inner' and 'outer' tensors.
    """
    tasks = []

    # Observational regime is one task
    obs_tensor, _ = sample_dict_to_tensor(dataset['observational'], order=order)
    for _ in range(batch_size):  # batch of tasks
        idx_in = np.random.choice(obs_tensor.size(0), batch_size, replace=False)
        idx_out = np.random.choice(obs_tensor.size(0), batch_size, replace=False)
        tasks.append({
            'inner': obs_tensor[idx_in],
            'outer': obs_tensor[idx_out],
        })

    # Each intervention regime is another task
    for var_name, sample_dict in dataset['interventional'].items():
        int_tensor, _ = sample_dict_to_tensor(sample_dict, order=order)
        for _ in range(batch_size):
            idx_in = np.random.choice(int_tensor.size(0), batch_size, replace=False)
            idx_out = np.random.choice(int_tensor.size(0), batch_size, replace=False)
            tasks.append({
                'inner': int_tensor[idx_in],
                'outer': int_tensor[idx_out],
            })

    return tasks


def train_model(config):
    torch.manual_seed(config.get('seed', 0))
    np.random.seed(config.get('seed', 0))

    # Generate a random categorical graph with ground-truth CPDs
    graph = generate_categorical_graph(
        num_vars=config['data']['num_vars'],
        min_categs=config['data']['num_categs'],
        max_categs=config['data']['num_categs'],
        edge_prob=config['data'].get('edge_prob', 0.5),
        connected=config['data'].get('connected', True),
        use_nn=True,
        deterministic=config['data'].get('deterministic', False),
        graph_func=get_graph_func(config['data']['graph_type']),
        seed=config.get('seed', 0),
        num_latents=config['data'].get('num_latents', 0)
    )

    # Build the dataset (observational + interventional)
    dataset = build_dataset(
        graph,
        num_obs=config['data']['num_obs'],
        num_int=config['data']['num_int']
    )

    # Convert one sample to get variable ordering
    sample0 = dataset['observational']
    _, order = sample_dict_to_tensor(sample0)

    # Instantiate the model
    mask = create_mask(graph, config['model']['structure'])
    model = create_model(config, mask=mask)
    device = torch.device(config.get('device', 'cpu'))
    model.to(device)

    obj_type = config['objective']['type']
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    if obj_type == 'pseudo_ll':
        optimizer = Adam(model.parameters(), lr=lr)
        # only observational data
        obs_tensor, _ = sample_dict_to_tensor(dataset['observational'], order=order)
        for epoch in range(epochs):
            idx = np.random.choice(obs_tensor.size(0), batch_size, replace=False)
            X = obs_tensor[idx].to(device)
            loss = pseudo_ll_loss(model, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] Pseudo-LL loss: {loss.item():.4f}")

    elif obj_type == 'maml':
        meta_optimizer = Adam(model.parameters(), lr=lr)
        inner_lr = config['objective']['inner_lr']
        inner_steps = config['objective']['inner_steps']
        first_order = config['objective'].get('first_order', True)
        for epoch in range(epochs):
            tasks = tasks_from_dataset(dataset, batch_size, order)
            meta_loss = maml_meta_update(
                model, tasks,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                first_order=first_order,
                meta_optimizer=meta_optimizer
            )
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] MAML meta-loss: {meta_loss:.4f}")

    else:
        raise ValueError(f"Unknown objective type: {obj_type}")

    return model, graph, order
