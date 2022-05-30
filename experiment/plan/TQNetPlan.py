import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from dataset import ModelNet40
from models import PointNetClassifier
from models.modules import TNet, QuaternionNet
from experiment.plan import Plan, ClassifierPlan
from utils import get_device


class TQNetPlan(Plan):
    def __init__(self, train_dataset: ModelNet40, test_dataset: ModelNet40,
                 output_path: str, device: str = get_device()):
        super(TQNetPlan, self).__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_path = output_path
        self.device = device

    def task(self, _):
        modes = ['vanilla', 'tnet', 'qnet']
        paths = [os.path.join(self.output_path, mode) for mode in modes]
        input_transforms = [nn.Identity(), TNet(3), QuaternionNet()]

        # Train
        for path, transform in zip(paths, input_transforms):
            model = self.make_model(transform)

            torch.manual_seed(999)
            torch.cuda.manual_seed(999)
            for param in model.parameters():
                if param.ndim > 1:
                    init.kaiming_normal_(param)

            optimizer = optim.Adam(model.parameters(), lr=1.0E-3, betas=(0.9, 0.999))
            plan = ClassifierPlan(model, optimizer,
                                  self.train_dataset, self.test_dataset,
                                  path, num_epochs=250)
            plan.execute()

        # Report
        result_paths = [os.listdir(path) for path in paths]
        result_filenames = [os.path.join(path, filenames[-1]) for path, filenames in zip(paths, result_paths)]
        bundles = [torch.load(filename) for filename in result_filenames]

        return {
            mode: bundle['test_accuracy'][-1] for
            mode, bundle in zip(modes, bundles)
        }

    def make_model(self, input_transform: nn.Module):
        model = PointNetClassifier(40)
        model.base.set_input_transform(input_transform)
        return model.to(self.device)
