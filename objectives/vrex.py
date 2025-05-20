import torch
from typing import Sequence, List, Tuple, Optional
from collections import OrderedDict
import time

from objectives.pseudo_ll import pseudo_ll_loss

__all__ = [
    "vrex_loss",
]

def vrex_loss(
    model: torch.nn.Module,
    env_batches: Sequence[torch.Tensor],
    lambda_penalty: float = 1.0,
    params: Optional[OrderedDict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Variation Risk Extrapolation (V-REx) loss (Krueger et al., 2021).

    The objective is
        L = Rbar + lambda Var(R_e),
    where R_e is the empirical risk in environment e.

    Parameters
    ----------
    model : A predictor returning per-sample probabilities as in other objectives.
    env_batches : One batch x_e per environment, each of shape [batch_size, num_vars].
    lambda_penalty : Weight for the variance penalty.
    params : Parameter dictionary to forward to the model.

    Returns
    -------
    total_loss : Scalar V-REx loss to be minimised.
    penalty : Variance component that enforces risk equality across environments.
    env_losses : Empirical risk per environment.
    """
    # t0 = time.perf_counter()
    # Compute all environment losses in parallel using torch.stack
    env_losses = torch.stack([pseudo_ll_loss(model, x_e, params=params) for x_e in env_batches])
    
    # Compute mean and variance in one go
    mean_risk = env_losses.mean()
    penalty = env_losses.var(unbiased=False)
    
    # Compute final loss
    total_loss = mean_risk + lambda_penalty * penalty
    # t1 = time.perf_counter()
    # print(f"V-REx computation time: {t1 - t0:.4f} seconds")
    return total_loss, penalty.detach(), [l.detach() for l in env_losses]
