# Data loader for multiple data
from datasets.example_dataset import get_loader as get_example_loader

def get_loader(cfg, phase):
    if cfg.data.name == 'example_dataset':
        return get_example_loader(cfg, phase)
    else:
        raise NotImplementedError