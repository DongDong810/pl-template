import torch,wandb,random
import torch.nn as nn
import os.path as osp
import numpy as np
from solvers.base_solver import BaseSolver
from datasets.example_dataset import get_loader

class Solver(BaseSolver):
    def __init__(self,cfg,net,criterion):
        super().__init__(cfg,net,criterion)
        self.cfg = cfg
        self.net = net
        self.criterion = criterion
    
    def training_step(self, batch_dict, batch_idx):
        x = batch_dict['x']
        y = batch_dict['y']

        # forward
        ret_dict = self.net(x)

        # loss
        loss_dict = self.criterion(ret_dict,batch_dict,'train')

        # log
        self.log_dict(loss_dict,
                      prog_bar=True,
                      rank_zero_only=True,)
        
        return loss_dict['train-total_loss']
    
    def validation_step(self, batch_dict, batch_idx):
        x = batch_dict['x']
        y = batch_dict['y']

        # forward
        ret_dict = self.net(x)

        # loss
        loss_dict = self.criterion(ret_dict,batch_dict,'val')

        # log
        self.log_dict(loss_dict,
                      prog_bar=True,
                      rank_zero_only=True,)
    
    def test_step(self, batch_dict, batch_idx):
        x = batch_dict['x']
        y = batch_dict['y']

        # forward
        y_hat = self.net(x)