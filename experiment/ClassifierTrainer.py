from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import ModelNet40
from dataset.transforms import RandomRotationUpAxis, RandomJittering
from models.modules import TransformLoss
from experiment import Trainer


class ClassifierTrainer(Trainer):
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 model: nn.Module, optimizer: optim.Optimizer):
        super().__init__(train_dataloader, test_dataloader, model, optimizer)
        self.model = model
        self.optimizer = optimizer

        self.augmentor = transforms.Compose([
            RandomRotationUpAxis(0, torch.pi * 2),
            RandomJittering(0, 0.02),
        ])

        self.ce_loss = nn.CrossEntropyLoss()
        self.t_loss = TransformLoss()

    def train(self):
        self.model.train()
        loss = 0
        for sample in tqdm(self.train_dataloader):
            pc = self.augmentor(sample['pc'])
            y = sample['label']

            input_transform, feat_transform, pred = self.model(pc)
            cls_loss = self.ce_loss(pred, y)
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

        # noinspection PyTypeChecker
        test_dataset: ModelNet40 = self.test_dataloader.dataset
        score = [0, len(test_dataset)]  # [correct, total]
        cls_scores = {value: [0, 0] for label, value in test_dataset.id2label.items()}

        for sample in tqdm(self.test_dataloader):
            pc = sample['pc']
            onehot = sample['label']

            _, _, output = self.model(pc)
            output_ids = torch.max(output, dim=1).indices

            ids = torch.max(onehot, dim=1).indices
            labels = [test_dataset.id2label[_id] for _id in ids.tolist()]

            matches = output_ids == ids
            for match, label in zip(matches, labels):
                if match:
                    score[0] += 1
                    cls_scores[label][0] += 1
                cls_scores[label][1] += 1

        score = score[0] / score[1] * 100
        for label in cls_scores.keys():
            s = cls_scores[label]
            cls_scores[label] = s[0] / s[1] * 100

        return {
            'overall_accuracy': score,
            'class_accuracy': cls_scores,
        }
