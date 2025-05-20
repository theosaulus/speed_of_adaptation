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
    outputs = model(x, params=params)            # (B,N,K)

    if var_indices is None:
        var_indices = range(outputs.size(1))

    outputs = outputs[:, var_indices, :]         # (B,|V|,K)
    logp = (outputs + 1e-12).log().flatten(0,1)  # (B*|V|,K)
    target = x[:, var_indices].reshape(-1)       # (B*|V|)
    return F.nll_loss(logp, target)