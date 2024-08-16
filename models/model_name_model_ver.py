import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class model_name(nn.Module):
    def __init__(self, cfg):
        super(model_name, self).__init__()
        self.cfg = cfg

        # Network architecture (layers)
        ################ write your code here ################
        # ex)
        self.fc1 = nn.Linear(10,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)  


    def forward(self, x) -> dict:
        """
        @param x: input (image as Tensor)
        @return: dictionary containing the output prediction of the network
        """

        ret_dict = {}

        ret_dict['y_hat'] = self.fc3(self.fc2(self.fc1(x)))  # save prediction

        return ret_dict
