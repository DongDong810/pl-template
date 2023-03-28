import torch
import torch.nn as nn

class model_name(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(10,10)

    def forward(self, x):
        ret_dict = {}
        ret_dict['y_hat'] = self.fc1(x)
        return ret_dict