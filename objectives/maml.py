import torch
import torch.nn.functional as F

from collections import OrderedDict
from objectives.maml_utils.gradient_based import gradient_update_parameters
from objectives.pseudo_ll import pseudo_ll_loss


def maml_loss(model, inner_batch, outer_batch, inner_lr, inner_steps=1, first_order=True):
    """
    Performs the inner-loop adaptation on inner_batch and computes the outer loss on outer_batch.

    Parameters:
      - model: a CausalCPDModel (MetaModule) to be adapted.
      - inner_batch: tensor (batch_size, num_vars) for inner (adaptation) loss.
      - outer_batch: tensor (batch_size, num_vars) for outer (meta) loss.
      - inner_lr: float, the step size for inner updates.
      - inner_steps: number of inner-loop steps.
      - first_order: whether to use the first-order approximation.
      
    Returns:
      outer_loss: the meta-loss computed using the adapted parameters.
    """
    params = OrderedDict(model.meta_named_parameters())
    for _ in range(inner_steps):
        loss_inner = pseudo_ll_loss(model, inner_batch, params=params)
        params = gradient_update_parameters(model,
                                              loss_inner,
                                              params=params,
                                              step_size=inner_lr,
                                              first_order=first_order)
    outer_loss = pseudo_ll_loss(model, outer_batch, params=params)
    return outer_loss

def maml_meta_update(model, tasks, inner_lr, inner_steps, first_order, meta_optimizer):
    """
    Given a list of tasks (each task is a dict with keys "inner" and "outer", each
    being a tensor of shape (batch_size, num_vars)), compute the average meta-loss,
    perform backpropagation, and step the meta-optimizer.

    Returns the meta-loss value.
    """
    meta_loss_total = 0.0
    for task in tasks:
        inner_batch = task["inner"]
        outer_batch = task["outer"]
        task_loss = maml_loss(model, inner_batch, outer_batch, inner_lr, inner_steps, first_order)
        meta_loss_total += task_loss
    meta_loss_total /= len(tasks)
    
    meta_optimizer.zero_grad()
    meta_loss_total.backward()
    meta_optimizer.step()
    
    return meta_loss_total.item()
