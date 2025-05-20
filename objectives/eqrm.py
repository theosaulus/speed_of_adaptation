
from typing import Sequence, List, Tuple, Optional
from collections import OrderedDict

import torch
from torch import Tensor

from objectives.pseudo_ll import pseudo_ll_loss

__all__ = ["eqrm_loss"]

def _positive_part(x: Tensor) -> Tensor:
    """max(x, 0) element-wise, but keeps grads."""
    return (x + x.abs()) * 0.5

def eqrm_loss(
    model,
    env_batches: Sequence[Tensor],
    tau: float = 0.75,
    lambda_penalty: float = 1.0,
    params: Optional[OrderedDict] = None,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    L = R̄  +  λ ·  1/E ∑_e (max( R_e − q_τ , 0 ))²
    where q_τ is the differentiable τ-quantile of the env risks {R_e}.
    """
    env_losses = torch.stack(
        [pseudo_ll_loss(model, x_e, params=params)
        for x_e in env_batches]
    )
    mean_risk = env_losses.mean()

    q_tau = torch.quantile(env_losses, tau)
    penalty = _positive_part(env_losses - q_tau).pow(2).mean()

    total_loss = mean_risk + lambda_penalty * penalty
    return total_loss, penalty.detach(), env_losses.detach().tolist()