import torch
import os
import os.path as osp
from torch.nn.parallel import DataParallel as DP
from utils.common import get_optimizer, get_scheduler
import lightning.pytorch as pl
import subprocess
from subprocess import DEVNULL, STDOUT

class BaseSolver(pl.LightningModule):
    def __init__(self,cfg,net,criterion):
        super().__init__()
        # basic config
        self.cfg = cfg
        self.net = net
        self.criterion = criterion
        self.epoch = -1

        # init best value & model dict
        self.best_value = {}
        for key_criteria in self.cfg.saver.monitor_keys:
            key,criteria = key_criteria.split('/')
            if criteria == 'l':
                self.best_value[key] = 987654321
            elif criteria == 'h':
                self.best_value[key] = -987654321

    def training_step(self, batch_dict, batch_idx):
        raise NotImplementedError()
    
    def validation_step(self, batch_dict, batch_idx):
        raise NotImplementedError()
    
    def test_step(self, batch_dict, batch_idx):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        # get optimizer and scheduler
        optimizer_mode = self.cfg.train.optimizer.mode
        optimizer = get_optimizer(opt_mode=optimizer_mode,
                                  net_params=self.net.parameters(),
                                  **(self.cfg.train.optimizer[optimizer_mode]))
        scheduler_mode = self.cfg.train.scheduler.mode
        scheduler = get_scheduler(sched_mode=scheduler_mode,
                                  optimizer=optimizer,
                                  **(self.cfg.train.scheduler[scheduler_mode]))
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val-total_loss'
            }
        }

    # def backup_result(self):
    #     if self.cfg.servername == 'dykim':
    #         return
        
    #     backup_sources = [self.cfg.path.ckpt_path, self.cfg.path.result_path]

    #     print('[Save]\tSaving results & ckpts...')
    #     for source in backup_sources:
    #         remote_target = self.cfg.backup_host + ':' + self.cfg.backup_target + source.replace('../','')
    #         remote_target = '/'.join(remote_target.split('/')[:-1])

    #         # make target dir in remote host
    #         os.system(f"ssh {self.cfg.backup_host} \'mkdir -p {os.path.join(self.cfg.backup_target,source.replace('../',''))}\'")
    #         # copy folders to remote
    #         subprocess.Popen(["rsync", '-r', source, remote_target], stdout=DEVNULL, stderr=STDOUT)
        
    #     return