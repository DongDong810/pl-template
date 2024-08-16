import torch, math

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch import Tensor

# Class for computing total loss (weighted sum of specific losses)
class MasterCriterion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.key_pairs = cfg.criterion.key_pairs # choose specific loss to use in this model (model_name.yml)
        
        self.mod_dict = {}
        self.mod_dict['l1_loss'] = L1_loss(cfg)
        self.mod_dict['cross_entropy'] = Cross_entropy(cfg)
        self.mod_dict['angular_loss'] = Angular_loss(cfg)
        #################### add more losses here ####################



    def forward(self, pred_dict, gt_dict, phase)-> dict:
        """
        Compute specfic losses -> weighted sum for total loss
        @param pred_dict: dictionary containing the output prediction of the network
        @param gt_dict: dictionary containing the ground truth
        @param phase: (train, val, test)
        @return: dictionary containing the total loss and specific losses
        """
        loss_dict = {}
        total_loss = 0
        
        # Total loss = weighted sum of losses (default.ymal)
        for loss_key in self.key_pairs:
            mod_key = self.cfg.criterion[loss_key].mod  # which loss?
            alpha = self.cfg.criterion[loss_key].alpha  # weight
            
            loss = self.mod_dict[mod_key](pred_dict, gt_dict) # compute specific loss
            loss_dict[f'{phase}-{loss_key}'] = loss
            total_loss += (alpha * loss)
        
        loss_dict[f'{phase}-total_loss'] = total_loss

        return loss_dict

########################### LOSS FUNCTIONS ###########################

class L1_loss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.L1 = nn.L1Loss()

    def forward(self, pred_dict, gt_dict):
        loss = self.L1(pred_dict['y_hat'], gt_dict['y'])

        return loss
    

class Cross_entropy(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred_dict, gt_dict):
        loss = self.cross_entropy(pred_dict['y_hat'], gt_dict['y'])

        return loss

class Angular_loss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def _compute_angular_loss(self, pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        cosine_similarity = F.cosine_similarity(pred, label, dim=1) # compute cosine similarity
        cosine_similarity = torch.clamp(cosine_similarity, min=-safe_v, max=safe_v) # clamp for safe value
        angle = torch.acos(cosine_similarity) # angle (Radians)
        angle = angle * (180.0 / math.pi) # angle (Degrees)
        return torch.mean(angle)

    def forward(self, pred_dict, gt_dict):
        loss = self._compute_angular_loss(pred_dict['y_hat'], gt_dict['y'])

        return loss
