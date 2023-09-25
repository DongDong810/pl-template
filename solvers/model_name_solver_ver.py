import torch,wandb,random
import torch.nn as nn
import os.path as osp
import numpy as np
from omegaconf import OmegaConf
from solvers.base_solver import BaseSolver
from datasets.example_dataset import get_loader

class Solver(BaseSolver):
    def __init__(self,net,loss,**cfg):
        super().__init__(net=net,loss=loss,**cfg)
        self.save_hyperparameters(ignore=['net','loss'])
        self.cfg = OmegaConf.create(cfg)
        self.net = net
        self.loss = loss
    
    def training_step(self, batch_dict, batch_idx):
        x = batch_dict['x']
        y = batch_dict['y']

        # forward
        ret_dict = self.net(x)

        # loss
        loss_dict = self.loss(ret_dict,batch_dict,'train')

        # log
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        
        return loss_dict['train-total_loss']
    
    def validation_step(self, batch_dict, batch_idx):
        x = batch_dict['x']
        y = batch_dict['y']

        # forward
        ret_dict = self.net(x)

        # loss
        loss_dict = self.loss(ret_dict,batch_dict,'val')

        # log
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
    
    def test_step(self, batch_dict, batch_idx):
        self.validation_step(batch_dict,batch_idx)