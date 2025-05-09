import numpy as np
import torch
from torch.optim import Adam

from data.dataset_creation import build_dataset, sample_dict_to_tensor
from data.graph_generation import generate_categorical_graph, get_graph_func
from data.utils import set_seed
from models.model_factory import create_model
from models.masks import create_mask
from objectives.pseudo_ll import pseudo_ll_loss
from objectives.maml import maml_meta_update


def tasks_from_dataset(dataset, batch_size, device, order):
    """
    Turn observational + interventional data into a list of tasks,
    each task being a dict with 'inner' and 'outer' tensors.
    """
    tasks = []

    # Observational regime is one task
    obs_tensor, _ = sample_dict_to_tensor(dataset['observational'], device, order=order)
    for _ in range(batch_size):  # batch of tasks
        B = batch_size if batch_size < obs_tensor.size(0) else obs_tensor.size(0)
        idx_in = np.random.choice(obs_tensor.size(0), B, replace=False)
        idx_out = np.random.choice(obs_tensor.size(0), B, replace=False)
        tasks.append({
            'inner': obs_tensor[idx_in],
            'outer': obs_tensor[idx_out],
        })

    # Each intervention regime is another task
    for var_name, sample_dict in dataset['interventional'].items():
        int_tensor, _ = sample_dict_to_tensor(sample_dict, device, order=order)
        for _ in range(batch_size):
            B = batch_size if batch_size < int_tensor.size(0) else int_tensor.size(0)
            idx_in = np.random.choice(int_tensor.size(0), B, replace=False)
            idx_out = np.random.choice(int_tensor.size(0), B, replace=False)
            tasks.append({
                'inner': int_tensor[idx_in],
                'outer': int_tensor[idx_out],
            })

    return tasks


def train_model(graph, dataset, order, config, device):
    set_seed(config.get('seed', 0))

    # Instantiate the model
    mask = create_mask(graph, config['model']['mask'], device=device)
    model = create_model(
        mask, 
        len(graph.variables),
        graph.variables[0].prob_dist.num_categs,
        config['model']['hidden_units'],
        device=device,
    )
    model.to(device)

    obj_type = config['objective']['type']
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    if obj_type == 'pseudo_ll':
        optimizer = Adam(model.parameters(), lr=lr)
        data_tensor, _ = sample_dict_to_tensor(dataset['observational'], device, order)
        for var_name, sample_dict in dataset['interventional'].items():
            int_tensor, _ = sample_dict_to_tensor(sample_dict, device, order)
            data_tensor = torch.cat((data_tensor, int_tensor), dim=0)

        for epoch in range(epochs):
            model.train()
            B = batch_size if batch_size < data_tensor.size(0) else data_tensor.size(0)
            idx = np.random.choice(data_tensor.size(0), B, replace=False)
            X = data_tensor[idx].to(device)
            loss = pseudo_ll_loss(model, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"[Epoch {epoch}] Pseudo-LL loss: {loss.item():.4f}")

    elif obj_type == 'maml':
        meta_optimizer = Adam(model.parameters(), lr=lr)
        inner_lr = config['objective']['inner_lr']
        inner_steps = config['objective']['inner_steps']
        first_order = config['objective'].get('first_order', True)
        for epoch in range(epochs):
            tasks = tasks_from_dataset(dataset, batch_size, device, order)
            meta_loss = maml_meta_update(
                model, tasks,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                first_order=first_order,
                meta_optimizer=meta_optimizer
            )
            if epoch % 200 == 0:
                print(f"[Epoch {epoch}] MAML meta-loss: {meta_loss:.4f}")

    else:
        raise ValueError(f"Unknown objective type: {obj_type}")

    return model
