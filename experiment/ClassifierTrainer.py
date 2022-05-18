from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.modules import TransformLoss
from experiment import Trainer


class ClassifierTrainer(Trainer):
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 model: nn.Module, optimizer: optim.Optimizer):
        super().__init__(train_dataloader, test_dataloader, model, optimizer)
        self.model = model
        self.optimizer = optimizer

        self.ce_loss = nn.CrossEntropyLoss()
        self.t_loss = TransformLoss()

    def train(self):
        self.model.train()
        loss = 0
        for sample in tqdm(self.train_dataloader):
            pc = sample['pc']
            y = sample['label']

            input_transform, feat_transform, pred = self.model(pc)
            cls_loss = self.ce_loss(y, pred)
            in_loss = self.t_loss(input_transform)
            feat_loss = self.t_loss(feat_transform)
            batch_loss = cls_loss + 1.0E-3 * (in_loss + feat_loss)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.clone().detach() / len(self.train_dataloader)
        return loss

    @torch.no_grad()
    def test(self):
        self.model.eval()
        total_loss = 0
        for sample in tqdm(self.test_dataloader):
            pc = sample['pc']
            y = sample['label']

            input_transform, feat_transform, pred = self.model(pc)
            loss = self.ce_loss(y, pred) + 1.0E-3 * (self.t_loss(input_transform) + self.t_loss(feat_transform))

            total_loss += loss.clone().detach()
        return total_loss / len(self.test_dataloader)
