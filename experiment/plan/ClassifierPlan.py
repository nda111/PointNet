import os
import pathlib

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.format as fmt
from dataset import ModelNet40
from experiment import ClassifierTrainer
from experiment.plan import Plan
from models import PointNetClassifier


class ClassifierPlan(Plan):
    def __init__(self, model: PointNetClassifier,
                 optimizer: optim.Optimizer,
                 train_dataset: ModelNet40, test_dataset: ModelNet40,
                 output_path: str,
                 num_epochs: int = 250):
        super(ClassifierPlan, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_path = output_path
        self.num_epochs = num_epochs

    def pre_task(self, variable):
        if not os.path.exists(self.output_path):
            path = pathlib.Path(self.output_path)
            path.mkdir(exist_ok=True, parents=True)

    def task(self, variable):
        train_dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=32)

        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        train_loss, test_loss = [], []

        trainer = ClassifierTrainer(train_dataloader, test_dataloader, model=self.model, optimizer=self.optimizer)
        for epoch in range(1, self.num_epochs + 1):
            print(f'epoch {epoch}, lr={self.optimizer.param_groups[0]["lr"]}')

            loss = trainer.train()
            train_loss.append(loss.item())
            scheduler.step()
            print(f'train_loss={loss.item()}')

            loss = trainer.test()
            test_loss.append(loss.item())
            print(f'test_loss={loss.item()}')
            print()

            if epoch % 5 == 0:
                bundle = {
                    'epochs': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                }

                file_name = f'{fmt.get_timestamp()}.pkl'
                torch.save(bundle, os.path.join(self.output_path, file_name))


