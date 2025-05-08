import torch.nn.functional as F

def pseudo_ll_loss(model, x, params=None, var_indices=None):
    """
    Computes the pseudo-loglikelihood loss over all CPD modules.
    Expects:
      - model: a CausalCPDModel whose forward(x, params) returns [batch_size, num_vars, output_dim].
      - x: a torch.LongTensor of shape [batch_size, num_vars].
      - params: an OrderedDict of parameters (if None, the model's own parameters are used).
      - var_indices: optional iterable of variable indices to include; if None, uses all variables.
    """
    outputs = model(x, params=params)  # (batch_size, num_vars, output_dim)
    loss = 0.0
    num_vars = outputs.shape[1]
    if var_indices is None:
        var_indices = range(num_vars)
    for i in var_indices:
        probs = outputs[:, i, :]
        logp = (probs + 1e-12).log()
        target = x[:, i].long()
        loss += F.nll_loss(logp, target)
    return loss / float(len(list(var_indices)))
