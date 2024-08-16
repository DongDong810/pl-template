import sys, os, random, torch, importlib, time, wandb
import torch.nn as nn
import torch.distributed as dist
import pytorch_lightning as pl

# Insert path for finding modules (order: . -> ../../ -> ..)
sys.path.insert(1, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('../../'))

from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchsummary import summary

# Functions & Classes from other files
from utils.setup import init_path_and_expname, get_callbacks, get_logger, get_trainer_args
from datasets.unified_loader import get_loader
from criterions.criterion import MasterCriterion

# Main - e.g. python main.py model.name={model_name} model.ver={model_ver} model.solver={solver_ver}
if __name__ == '__main__':
    # 1. Import default config file
    cfg = OmegaConf.load(f'../configs/default.yaml')
    # Read from command line (model.name / model.ver / model.solver)
    cfg_cmd = OmegaConf.from_cli()  # return dictionary object
    """
    {
        "model": {
            "name": "model_name",
            "ver": "model_ver",
            "solver": "solver_ver"
        }
    }
    """
    # 2. Merge model specific config file
    if "model" in cfg_cmd  and 'name' in cfg_cmd.model: # check key (model.name)
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f'../configs/{cfg_cmd.model.name}.yaml'))  # model from command line
    else:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(f'../configs/{cfg.model.name}.yaml'))  # model from default.yaml
    # Merge cfg with command line
    cfg = OmegaConf.merge(cfg, cfg_cmd)  # default.yaml + model_name.yaml + command line


    # Config for path, exp_name, random seed
    init_path_and_expname(cfg)  # only done in master process (setup.py)
    pl.seed_everything(cfg.seed)


    # Dataloader
    dataloader = {
        'train': get_loader(cfg, 'train') if cfg.mode == 'train' else None,
        'valid': get_loader(cfg, 'valid') if cfg.mode == 'train' else None,
        'test' : get_loader(cfg, 'test') if cfg.mode == 'test' else None
    }


    # Dynamic Model module import
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')  # import models/{model_name}_{model_ver}.py
    network_class = getattr(network_mod, cfg.model.name)  # get "{model_name}" class
    network = network_class(cfg)  # make instance with the class


    # Loss
    loss = MasterCriterion(cfg)  # make instance from criterions/criterion.py


    # Dynamic Solver module import
    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')  # import solvers/{model_name}_{solver_ver}.py
    solver_class = getattr(solver_mod, 'Solver')  # get "Solver" class
    solver = solver_class(net=network,
                          loss=loss,
                          **(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)))  # make instance with the class


    # Summary of model
    # summary(network, (1, 28, 28))  # (in_channels, input image height, input image width)


    # Init trainer
    trainer_args = get_trainer_args(cfg)  # dictionary of train args (setup.py)
    trainer = pl.Trainer(**trainer_args)  # make trainer with train_args


    # Automatically train or test
    if cfg.mode == 'train':
        trainer.fit(
            model=solver,
            train_dataloaders=dataloader['train'],
            val_dataloaders=dataloader['valid'],
            ckpt_path=cfg.load.ckpt_path if cfg.load.load_state else None
        )
    
    elif cfg.mode == 'test':
        trainer.test(
            model=solver,
            dataloaders=dataloader['test']
        )
