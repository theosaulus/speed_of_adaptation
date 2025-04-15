import torch.nn.functional as F

def pseudo_ll_loss(model, x, params=None):
    """
    Computes the pseudo-loglikelihood loss over all CPD modules.
    Expects:
      - model: a CausalCPDModel whose forward(x, params) returns [batch_size, num_vars, output_dim].
      - x: a torch.LongTensor of shape [batch_size, num_vars].
      - params: an OrderedDict of parameters (if None, the model's own parameters are used).
    """
    outputs = model(x, params=params)  # (batch_size, num_vars, output_dim)
    loss = 0.0
    num_vars = outputs.shape[1]
    for i in range(num_vars):
        logits = outputs[:, i, :]
        target = x[:, i].long()
        loss += F.cross_entropy(logits, target)
    return loss / num_vars
