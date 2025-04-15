import torch 
import torch.nn as nn
import torch.nn.functional as F


class CPDModel(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, mask):
        super(CPDModel, self).__init__()
        self.mask = mask  # mask is a tensor of shape [input_dim] (or [input_dim, output_dim] if needed)
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_dim)
    
    def forward(self, x):
        x_masked = x * self.mask
        h = F.leaky_relu(self.fc1(x_masked), negative_slope=0.1)
        logits = self.fc2(h)
        return logits # to be used with softmax


class CausalCPDModel(nn.Module): 
    """
    Combines multiple CPD models, each predicting for a specific variable. 
    Input: (batch_size, num_vars). 
    Output: (batch_size, num_vars, output_dim).
    """
    def __init__(self, cpd_models):
        super(CausalCPDModel, self).__init__()
        self.cpd_models = nn.ModuleList(cpd_models)

    def forward(self, x):
        # x: (batch_size, num_vars)
        outputs = []
        for model in self.cpd_models:
            logits = model(x)  # (batch_size, output_dim)
            outputs.append(logits.unsqueeze(1))  # unsqueeze to add the variable dimension
        outputs_cat = torch.cat(outputs, dim=1)  # Concatenate along the variable dimension
        return outputs_cat # (batch_size, num_vars, output_dim)


def create_model(config, mask): 
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
    num_vars = config['data']['num_vars']
    output_dim = config['data']['output_dim']
    hidden_units = config['model']['hidden_units']
    input_dim = num_vars

    # Convert mask to torch.Tensor if necessary and ensure correct type
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32)
    if mask.shape != (num_vars, num_vars):
        raise ValueError(f"Expected mask shape ({num_vars}, {num_vars}), but got {mask.shape}")

    cpdm_list = []
    # For each variable, extract corresponding mask vector (column of the mask)
    for i in range(num_vars):
        mask_i = mask[:, i]  # Shape: (num_vars,)
        cpdm = CPDModel(input_dim=input_dim, hidden_units=hidden_units, output_dim=output_dim, mask=mask_i)
        cpdm_list.append(cpdm)

    model = CausalCPDModel(cpdm_list)
    return model