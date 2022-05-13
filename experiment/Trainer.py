from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class Trainer(ABC):
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader, *args):
        self.args = args
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    @abstractmethod
    def train(self):
        pass

    @torch.no_grad()
    @abstractmethod
    def test(self):
        pass
