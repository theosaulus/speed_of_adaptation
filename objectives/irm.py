import torch
from torch import Tensor
from torch.autograd import grad
from typing import Sequence, List, Tuple, Optional
from collections import OrderedDict

from objectives.pseudo_ll import pseudo_ll_loss

__all__ = [
    "irm_penalty",
    "irm_loss",
]

def irm_penalty(risk: Tensor, dummy: Tensor) -> Tensor:
    """Compute the IRMv1 penalty ‖nabla_dummy R‖².

    Parameters
    risk : Scalar empirical risk for one environment (must retain graph).
    dummy : Scalar dummy parameter that scales the predictor.
    
    Returns
    Squared gradient norm w.r.t. dummy.
    """
    grad_dummy = grad(risk, [dummy], create_graph=True)[0]
    return grad_dummy.pow(2)


class _ScaledModel(torch.nn.Module):
    """
    Wrap an arbitrary model so that its outputs are *multiplied* by the
    dummy scalar.  The dummy lives outside the parameter list, so the optimiser
    never updates it, but it still participates in autograd.
    """
    def __init__(self, base_model: torch.nn.Module, dummy: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.dummy = dummy

    def forward(self, x, params = None) -> Tensor:
        probs = self.base_model(x, params=params).clamp_min(1e-12)
        logits = probs.log()
        scaled_logits = logits * self.dummy
        return torch.softmax(scaled_logits, dim=-1)


def irm_loss(
    model: torch.nn.Module,
    env_batches: Sequence[Tensor],
    lambda_penalty: float = 1.0,
    params: Optional[OrderedDict] = None,
    dummy_init: float = 1.0,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Invariant Risk Minimisation (IRMv1) loss suitable for gradient descent.

    Implements the objective
        L = Rbar + lambda · sum_e ||nabla_dummy R_e||**2
    where Rbar is the mean empirical risk across environments and the
    penalty encourages invariance of the optimal classifier across them.

    Parameters
    ----------
    model : The predictive model. Must expose forward(x, params=None) like
            the other objectives (cf. pseudo_ll).
    env_batches :  List/tuple of batches x_e, one for each environment. Each tensor is
                   shaped [batch_size, num_vars].
    lambda_penalty : Weight of the IRM penalty term.
    params : Explicit parameter dictionary (e.g. during meta-learning). If None, the 
             model's own parameters are used.
    dummy_init : Initial value of the scaling dummy parameter.

    Returns
    -------
    total_loss : The scalar IRM loss.
    penalty : The invariance penalty term.
    env_losses : List of per-environment empirical risks.
    """
    device = env_batches[0].device
    dummy  = torch.tensor(dummy_init, requires_grad=True, device=device)
    scaled_model = _ScaledModel(model, dummy)

    env_losses = []
    penalties = []

    for x_e in env_batches:
        risk_e = pseudo_ll_loss(scaled_model, x_e, params=params)
        env_losses.append(risk_e)
        penalties.append(irm_penalty(risk_e, scaled_model.dummy))

    mean_risk = torch.stack(env_losses).mean()
    penalty = torch.stack(penalties).mean()
    total_loss = mean_risk + lambda_penalty * penalty

    return total_loss, penalty.detach(), [l.detach() for l in env_losses]

