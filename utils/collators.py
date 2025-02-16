# collators.py
import torch
import pandas as pd

class CollateObject:
    def __init__(self, input_norm, target_norm, device):
        self.input_norm = input_norm
        self.target_norm = target_norm
        self.device = device
    
    def __call__(self, data):
        if len(data[0]) == 2:  # If we receive (x, y)
            x = pd.concat([sample[0] for sample in data], axis=0)
            y = pd.concat([sample[1] for sample in data], axis=0)
            desc = None
            n_to_fill = None
        else:  # If we receive (x, desc, n_to_fill, y)
            x = pd.concat([sample[0] for sample in data], axis=0)
            desc = list(pd.concat([sample[1] for sample in data], axis=0))
            n_to_fill = list(pd.concat([sample[2].iloc[0] for sample in data], axis=0))
            y = pd.concat([sample[3] for sample in data], axis=0)
    
        x = torch.tensor(self.input_norm.transform(x),
                         device=self.device,
                         requires_grad=True).to(torch.float32)
        y = torch.tensor(self.target_norm.transform(y),
                         device=self.device).to(torch.float32)
        if desc is not None:
            return x, desc, n_to_fill, y
        return x, y
