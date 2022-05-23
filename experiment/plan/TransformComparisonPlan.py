import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ModelNet40
from models import PointNetClassifier
from experiment.plan import Plan, ClassifierPlan
from models.modules import TNet, QuaternionNet
from utils import get_device, clone_object


class TransformComparisonPlan(Plan):
    def __init__(self, train_dataset: ModelNet40, test_dataset: ModelNet40, output_path: str,
                 device: str = get_device()):
        super(TransformComparisonPlan, self).__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_path = output_path
        self.device = device

    def task(self, variable):
        seed = 999
        paths = [os.path.join(self.output_path, dir_name) for dir_name in ('vanilla', 'tnet', 'qnet')]

        # Train vanilla model
        model = self.make_vanilla_model()
        optimizer = optim.Adam(model.parameters(), lr=1.0E-3)
        trainer = ClassifierPlan(model, optimizer,
                                 self.train_dataset, self.test_dataset,
                                 paths[0], num_epochs=250)
        TransformComparisonPlan.reset_random_seed(seed)
        trainer.execute()

        # Make qnet model, optim
        vanilla_state_dict = TransformComparisonPlan.clone_state_dict(model.state_dict())
        model_qnet = self.make_vanilla_model(vanilla_state_dict)
        qnet = QuaternionNet().to(self.device)
        model_qnet.base.set_input_transform(qnet)
        optimizer_qnet = optim.Adam(qnet.parameters(), lr=1.0E-3)

        # Turn vanilla model into tnet model, make its optim
        model_tnet = model
        tnet = TNet(3).to(self.device)
        model_tnet.base.set_input_transform(tnet)
        optimizer_tnet = optim.Adam(tnet.parameters(), lr=1.0E-3)

        # Train tnet model
        trainer = ClassifierPlan(model_tnet, optimizer_tnet,
                                 self.train_dataset, self.test_dataset,
                                 paths[1], num_epochs=100)
        TransformComparisonPlan.reset_random_seed(seed)
        trainer.execute()

        # Train qnet model
        trainer = ClassifierPlan(model_qnet, optimizer_qnet,
                                 self.train_dataset, self.test_dataset,
                                 paths[2], num_epochs=100)
        TransformComparisonPlan.reset_random_seed(seed)
        trainer.execute()

        result_paths = [os.listdir(path) for path in paths]
        result_filenames = [os.path.join(path, filenames[-1]) for path, filenames in zip(paths, result_paths)]
        vanilla_bundle, tnet_bundle, qnet_bundle = [torch.load(filename) for filename in result_filenames]

        return {
            'vanilla': vanilla_bundle['test_accuracy'][-1],
            'tnet': tnet_bundle['test_accuracy'][-1],
            'qnet': qnet_bundle['test_accuracy'][-1],
        }

    def make_vanilla_model(self, state_dict=None):
        model = PointNetClassifier(40)
        model.base.set_input_transform(nn.Identity())
        model.to(self.device)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)

        return model.to(self.device)

    @staticmethod
    def clone_state_dict(state_dict):
        filename = f'{fmt.get_timestamp()}.tmp'
        torch.save(state_dict, filename)
        clone = torch.load(filename)
        os.remove(filename)
        return clone

    @staticmethod
    def reset_random_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
