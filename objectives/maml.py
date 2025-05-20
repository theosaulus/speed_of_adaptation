import torch
import torch.nn.functional as F
import time

from collections import OrderedDict
from objectives.maml_utils.gradient_based import gradient_update_parameters
from objectives.pseudo_ll import pseudo_ll_loss
from torch.func import vmap, functional_call, grad
from torch.amp import autocast, GradScaler

_scaler = GradScaler('cuda')


def _inner_adapt(model, params, buffers, inner_data, inner_lr, inner_steps):
    """Per-task inner loop returning *adapted* parameters.

    Args
    ----
    model       : the MetaModule (no side-effects, we use functional_call)
    params      : OrderedDict(str -> Tensor) - *shared* base params
    buffers     : dict(str -> Tensor)
    inner_data  : Tensor (S, num_vars) - support examples for 1 task
    inner_lr    : float - inner-loop SGD step size
    inner_steps : int   - #inner updates
    """

    def loss_fn(p, d):
        return pseudo_ll_loss(model, d, params=p)

    p = params
    for _ in range(inner_steps):
        grads = grad(loss_fn)(p, inner_data)  # OrderedDict with same keys
        p = {k: p[k] - inner_lr * grads[k] for k in p}
    return p  # adapted OrderedDict


def maml_loss(model, inner_batch, outer_batch, inner_lr, inner_steps=1):
    """Vectorised MAML loss for a *batch of tasks*.

    Parameters
    ----------
    model        : CausalCPDModel (inherits from MetaModule)
    inner_batch  : Tensor (T, S, num_vars) - support sets for T tasks
    outer_batch  : Tensor (T, O, num_vars) - query sets  for T tasks
    inner_lr     : float - inner SGD step size
    inner_steps  : int   - how many inner updates

    Returns
    -------
    torch.Tensor scalar - mean outer loss over the T tasks.
    """
    params = OrderedDict(model.meta_named_parameters())
    buffers = dict(model.named_buffers())

    # Adapt parameters for each task (vectorised over task dimension)
    adapted_params = vmap(
        lambda data: _inner_adapt(model, params, buffers, data, inner_lr, inner_steps)
    )(inner_batch)

    # Compute the outer losses in parallel
    outer_losses = vmap(
        lambda p, d: pseudo_ll_loss(model, d, params=p)
    )(adapted_params, outer_batch)

    return outer_losses.mean()


def maml_meta_update(
    model,
    tasks,
    inner_lr,
    inner_steps,
    meta_optimizer,
    **kwargs  # Add this to handle any additional kwargs
):
    """Perform a *single* meta-update step.

    Parameters
    ----------
    model          : CausalCPDModel
    tasks          : list of dicts with keys 'inner', 'outer' - each a Tensor
    inner_lr       : float - step size for inner loop
    inner_steps    : int   - #inner updates per task
    meta_optimizer : torch.optim.Optimizer - drives the meta-parameters
    **kwargs       : additional keyword arguments (ignored)

    Returns
    -------
    float - scalar value of the meta loss (Python number).
    """
    # t0 = time.perf_counter()

    # Stack the per-task tensors : shape (T, …) expected by maml_loss
    inner_batch = torch.stack([t["inner"] for t in tasks])
    outer_batch = torch.stack([t["outer"] for t in tasks])

    meta_optimizer.zero_grad(set_to_none=True)

    # Forward + loss under AMP
    with autocast('cuda'):
        meta_loss = maml_loss(model, inner_batch, outer_batch, inner_lr, inner_steps)

    # Backward with gradient scaling (handles mixed precision safely)
    _scaler.scale(meta_loss).backward()
    _scaler.step(meta_optimizer)
    _scaler.update()

    # t1 = time.perf_counter()
    # print(f"Meta-update total time: {t1 - t0:.4f} s")

    return meta_loss.item()