import os, sys, torch

import numpy as np
import torchvision.transforms as T

# sys.path.insert(1, os.path.abspath('..'))
from torch.utils.data import Dataset, DataLoader


class ExampleDataset(Dataset):
    def __init__(self, cfg, phase):
        super(ExampleDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.data = self._load_data()
        self.labels = self._load_labels()
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        @param idx: index of the data
        @return: dictionary with the data (input data & GT label)
        """
        batch_dict = {}

        # get data
        batch_dict['x'] = self.data[idx]

        # get label (only for train and valid phase)
        if self.phase in ['train', 'valid']:
            batch_dict['y'] = self.labels[idx]
        else:
             batch_dict['y'] = None

        # transform if needed
        if self.transform:
            batch_dict = self.transform(batch_dict)

        return batch_dict

    ############### Implement below functions ############

    def _load_data(self):
        # ex)
        return torch.randn(10000, 10)

    def _load_labels(self):
        # ex)
        return torch.randint(0, 10, (10000, 10))

    def _get_transform(self):
        # ex)
        if self.phase == 'train':
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        elif self.phase in ['valid', 'test']:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = None

        return transform


# Function for data loader
def get_loader(cfg, phase):
    """
    @param cfg: config file
    @param phase: ['train', 'valid', 'test']
    @return: data loader
    """
    dataset = ExampleDataset(cfg, phase)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True, # for GPU
    )
    return loader
