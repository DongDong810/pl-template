import sys,os,random,torch,importlib
import torch.nn as nn
import wandb
from omegaconf import OmegaConf

sys.path.insert(1,os.path.abspath('..'))
sys.path.insert(1,os.path.abspath('../../'))
from utils.common import set_random_seed,init_experiment,get_callbacks,get_logger
from datasets.example_dataset import get_loader
from criterions.criterion import MasterCriterion
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

def main(cfg):
    # Set random seed
    # if cfg.mode == 'test' and cfg.random_seed is None:
    #     cfg.random_seed = 0
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed)

    # Dataloader
    dataloader = {
        'train': get_loader(cfg,'train'),
        'valid': get_loader(cfg,'valid'),
        'test' : get_loader(cfg,'test')
    }
    
    # Dynamic Model module import
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod,cfg.model.name)
    network = network_class(cfg)
    
    # Loss
    criterion = MasterCriterion(cfg)

    # Dynamic Solver module import
    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')
    solver_class = getattr(solver_mod,'Solver')
    solver = solver_class(cfg,network,criterion)
    
    # Load Network if ckpt_path is given
    if cfg.load.ckpt_path is not None:
        solver.load_from_checkpoint(cfg.load.ckpt_path)

    # get custom callback list for trainer
    callbacks = get_callbacks(cfg)
    # get logger
    logger = get_logger(cfg)

    # Let's go!
    trainer = pl.Trainer(
        devices=cfg.devices,
        accelerator="auto",
        logger=logger,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=callbacks,
    )

    if cfg.mode == 'train':
        trainer.fit(model=solver,
                    train_dataloaders=dataloader['train'],
                    val_dataloaders=dataloader['valid'],
                    ckpt_path=cfg.load.ckpt_path if cfg.load.load_state else None)
    
    elif cfg.mode == 'test':
        trainer.test(model=solver,
                     test_dataloaders=dataloader['test'])

if __name__ == '__main__':
    # import default config file
    cfg = OmegaConf.merge(OmegaConf.load(f'../configs/default.yaml'),OmegaConf.load('../configs/env.yaml'))
    # read from command line
    cfg_cmd = OmegaConf.from_cli()
    # merge model specific config file
    if "model" in cfg_cmd  and 'name' in cfg_cmd.model:
        cfg = OmegaConf.merge(cfg,OmegaConf.load(f'../configs/{cfg_cmd.model.name}.yaml'))
    else:
        cfg = OmegaConf.merge(cfg,OmegaConf.load(f'../configs/{cfg.model.name}.yaml'))
    # merge cfg from command line
    cfg = OmegaConf.merge(cfg,cfg_cmd)

    # Path configuration & generation
    init_experiment(cfg)

    main(cfg)