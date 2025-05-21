
from typing import Sequence, List, Tuple, Optional
from collections import OrderedDict

import math
import torch
from torch import Tensor

from objectives.pseudo_ll import pseudo_ll_loss

__all__ = ["eqrm_loss"]

def _normal_icdf(u: float, device, dtype) -> Tensor:
    """
    Inverse CDF of the standard normal, returned as a 0-D tensor on the
    right device / dtype so it stays in the computation graph.
    """
    u_tensor = torch.tensor(u, device=device, dtype=dtype)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u_tensor - 1.0)

def eqrm_loss(
    model,
    env_batches: Sequence[Tensor],
    tau: float = 0.75,
    icdf: str = "normal",
    params: Optional[OrderedDict] = None,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    EQRM
    """
    env_losses = torch.stack(
        [pseudo_ll_loss(model, x_e, params=params)
        for x_e in env_batches]
    )

    if icdf == "normal":
        mu = env_losses.mean()
        sigma = env_losses.std(unbiased=False) + 1e-12
        z_tau = _normal_icdf(tau, env_losses.device, env_losses.dtype)
        q_tau = mu + z_tau * sigma
        total_loss = q_tau
        return total_loss, env_losses.detach().tolist()
    
    elif icdf == "none":
        total_loss = torch.quantile(env_losses, tau)
        return total_loss, env_losses.detach().tolist()

    else:
        raise ValueError(f"Unknown ICDF type: {icdf}. Supported types are 'normal' and 'none'.")