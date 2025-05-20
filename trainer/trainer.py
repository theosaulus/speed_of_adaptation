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
from objectives.irm import irm_loss
from objectives.vrex import vrex_loss
from objectives.eqrm import eqrm_loss

def tasks_from_dataset(dataset, batch_size, device, order,
                       k_inner: int = 20) -> list[dict[str, torch.Tensor]]:
    """
    Build a list of tasks for MAML.

    Args
    ----
    dataset     : dict with 'observational' and 'interventional' keys
    batch_size  : number of tasks per regime (obs + each intervention)
    device      : torch device
    order       : list/tuple giving variable order for sample_dict_to_tensor
    k_inner     : #support examples per task (≥1)

    Returns
    -------
    tasks       : list of dicts, each with keys 'inner' and 'outer'
    """
    tasks = []

    def split_indices(num_rows: int, k_in: int, num_tasks: int):
        """
        Draw  (k_in + 1) * num_tasks  unique rows without replacement
        if possible; otherwise fall back to with-replacement sampling.
        """
        needed = (k_in + 1) * num_tasks
        if num_rows >= needed:
            idx = torch.randperm(num_rows, device=device)[:needed]
        else:
            idx = torch.randint(0, num_rows, (needed,), device=device)
            if k_in == 1:
                inner = idx[:num_tasks]
                outer = idx[num_tasks:]
                dup_mask = outer == inner
                while dup_mask.any():
                    outer[dup_mask] = torch.randint(
                        0, num_rows, (dup_mask.sum().item(),), device=device)
                    dup_mask = outer == inner
                idx[num_tasks:] = outer
        return idx.view(num_tasks, k_in + 1)

    # observational regime
    obs_tensor, _ = sample_dict_to_tensor(
        dataset["observational"], device, order
    )
    idx_mat = split_indices(len(obs_tensor), k_inner, batch_size)
    for row in idx_mat:
        inner_idx, outer_idx = row[:-1], row[-1]
        tasks.append(
            {
                "inner": obs_tensor[inner_idx], # (k_inner, num_vars)
                "outer": obs_tensor[outer_idx : outer_idx + 1], # (1, num_vars)
            }
        )

    # interventional regimes
    for var_name, sample_dict in dataset["interventional"].items():
        int_tensor, _ = sample_dict_to_tensor(sample_dict, device, order)
        idx_mat = split_indices(len(int_tensor), k_inner, batch_size)
        for row in idx_mat:
            inner_idx, outer_idx = row[:-1], row[-1]
            tasks.append(
                {
                    "inner": int_tensor[inner_idx],
                    "outer": int_tensor[outer_idx : outer_idx + 1],
                }
            )

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
    # model = torch.compile(model)

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

    elif obj_type == 'maml': # Careful with the MAML objective: use lower batch-size
        meta_optimizer = Adam(model.parameters(), lr=lr)
        inner_lr = config['objective']['inner_lr']
        inner_steps = config['objective']['inner_steps']
        first_order = config['objective'].get('maml_first_order', True)
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

    elif obj_type == 'irm':
        lambda_pen = config['objective'].get('lambda_penalty', 1.0)
        optimizer  = Adam(model.parameters(), lr=lr)

        env_batches = []
        data_tensor, _ = sample_dict_to_tensor(dataset['observational'], device, order)
        env_batches.append(data_tensor)
        for sample_dict in dataset['interventional'].values():
            int_tensor, _ = sample_dict_to_tensor(sample_dict, device, order)
            env_batches.append(int_tensor)

        for epoch in range(epochs):
            model.train()
            loss, penalty, env_losses = irm_loss(
                model, env_batches, lambda_penalty=lambda_pen
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"[Epoch {epoch}] IRM loss: {loss.item():.4f} (pen {penalty.item():.4f})")
    
    elif obj_type == 'vrex':
        lambda_pen = config['objective'].get('lambda_penalty', 1.0)
        optimizer  = Adam(model.parameters(), lr=lr)

        env_batches = []
        data_tensor, _ = sample_dict_to_tensor(dataset['observational'], device, order)
        env_batches.append(data_tensor)
        for sample_dict in dataset['interventional'].values():
            int_tensor, _ = sample_dict_to_tensor(sample_dict, device, order)
            env_batches.append(int_tensor)

        for epoch in range(epochs):
            model.train()
            loss, penalty, env_losses = vrex_loss(
                model, env_batches, lambda_penalty=lambda_pen
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"[Epoch {epoch}] V-REx loss: {loss.item():.4f} (var {penalty.item():.4f})")

    elif obj_type == 'eqrm':
        lambda_pen = config['objective'].get('lambda_penalty', 1.0)
        tau = config['objective'].get('tau', 0.75)
        optimizer = Adam(model.parameters(), lr=lr)

        env_batches = []
        obs_tensor, _ = sample_dict_to_tensor(dataset['observational'], device, order)
        env_batches.append(obs_tensor)
        for sample_dict in dataset['interventional'].values():
            int_tensor, _ = sample_dict_to_tensor(sample_dict, device, order)
            env_batches.append(int_tensor)

        for epoch in range(epochs):
            loss, penalty, _ = eqrm_loss(
                model, env_batches, tau=tau, lambda_penalty=lambda_pen
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"[Epoch {epoch}] EQRM loss: {loss.item():.4f} (penalty {penalty.item():.4f})")

    else:
        raise ValueError(f"Unknown objective type: {obj_type}")

    return model
