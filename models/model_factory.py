import torch 
import torch.nn as nn
import torch.nn.functional as F

from objectives.maml_utils.module import MetaModule


class CPDModel(nn.Module):
    def __init__(self, num_vars, num_categs, hidden_units, mask):
        super(CPDModel, self).__init__()
        self.input_dim = num_vars * num_categs
        self.hidden_units = hidden_units
        self.output_dim = num_categs
        
        self.fc1 = nn.Linear(self.input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, self.output_dim)

        mask_col = mask.view(num_vars, 1) # mask is a tensor of shape [num_vars]
        mask_col = mask_col.expand(num_vars, num_categs) # expand to [num_vars, num_categs]
        self.register_buffer('mask_flat', mask_col.reshape(-1)) # do not track gradients

    def forward(self, x, params=None):
        # x: (batch_size, num_vars)
        batch_size = x.size(0)
        x_onehot = F.one_hot(x.long(), num_classes=self.output_dim).float() # (batch_size, num_vars, num_categs)
        x_flat = x_onehot.view(batch_size, -1)  # (batch_size, num_vars * num_categs)
        x_masked = x_flat * self.mask_flat[None, :]

        # Load parameters if provided
        if params is not None:
            w1 = params['fc1.weight']
            b1 = params['fc1.bias']
            w2 = params['fc2.weight']
            b2 = params['fc2.bias']
            h = F.leaky_relu(F.linear(x_masked, w1, b1), negative_slope=0.1)
            logits = F.linear(h, w2, b2)
        else:
            h = F.leaky_relu(self.fc1(x_masked), negative_slope=0.1)
            logits = self.fc2(h)
        probs = F.softmax(logits, dim=-1)  # (batch_size, output_dim)
        return probs

class CausalCPDModel(MetaModule): 
    """
    Combines multiple CPD models, each predicting for a specific variable. 
    Input: (batch_size, num_vars). 
    Output: (batch_size, num_vars, output_dim).
    """
    def __init__(self, cpd_models):
        super(CausalCPDModel, self).__init__()
        self.cpd_models = nn.ModuleList(cpd_models)

    def forward(self, x, params=None):
        # x: (batch_size, num_vars)
        outputs = []
        for i, model in enumerate(self.cpd_models):
            if params is not None:
                sub_params = self.get_subdict(params, f'cpd_models.{i}')
            else: 
                sub_params = None
            logits = model(x, params=sub_params)  # (batch_size, output_dim)
            outputs.append(logits.unsqueeze(1))  # unsqueeze to add the variable dimension
        outputs_cat = torch.cat(outputs, dim=1)  # Concatenate along the variable dimension
        return outputs_cat # (batch_size, num_vars, output_dim)


def create_model(mask, num_vars, output_dim, hidden_units, device): 
    """
    Creates a stack of CPD models for each variable in the dataset.
    The config must contain:
    - config['data']['num_vars']: Number of variables in the dataset.
    - config['data']['output_dim']: Number of output dimensions (e.g. categories) for each CPD.
    - config['model']['hidden_units']: Number of hidden units for the CPD MLPs.

    The mask must be an array (or torch.Tensor) where, 
    for variable i, the allowed input predictors are defined by mask[:, i]. 

    Parameters:
    config : dict
    mask : array-like or torch.Tensor (num_vars, num_vars)
    
    Returns:
    model : torch.nn.Module
        A module representing the full set of CPD models. Its forward pass accepts an input tensor
        of shape (batch_size, num_vars) and returns an output tensor of shape (batch_size, num_vars, output_dim).
    """
    input_dim = num_vars * output_dim

    # Convert mask to torch.Tensor if necessary and ensure correct type
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
    if mask.shape != (num_vars, num_vars):
        raise ValueError(f"Expected mask shape ({num_vars}, {num_vars}), but got {mask.shape}")

    cpdm_list = []
    # For each variable, extract corresponding mask vector (column of the mask)
    for i in range(num_vars):
        mask_i = mask[:, i]  # Shape: (num_vars,)
        cpdm = CPDModel(
            num_vars=num_vars, 
            num_categs=output_dim, 
            hidden_units=hidden_units, 
            mask=mask_i)
        cpdm_list.append(cpdm)

    model = CausalCPDModel(cpdm_list)
    return model