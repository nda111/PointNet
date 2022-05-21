import os

import torch.optim as optim

from dataset import ModelNet40
from models import PointNetClassifier
from experiment.plan import Plan, ClassifierPlan
from models.modules import TNet, QuaternionNet


class TransformComparisonPlan(Plan):
    def __init__(self, train_dataset: ModelNet40, test_dataset: ModelNet40, output_path: str):
        super(TransformComparisonPlan, self).__init__()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_path = output_path

    def task(self, variable):
        paths = [os.path.join(self.output_path, dir_name) for dir_name in ('vanilla', 'tnet', 'qnet')]
        model = PointNetClassifier(40)

        optimizer = optim.Adam(model.parameters(), lr=1.0E-3)
        trainer = ClassifierPlan(model, optimizer, self.train_dataset, self.test_dataset, paths[0], num_epochs=2)
        trainer.execute()

        model_tnet = model
        tnet = TNet(3)
        model_tnet.base.set_input_transform(tnet)
        optimizer_tnet = optim.Adam(tnet.parameters(), lr=1.0E-3)
        trainer = ClassifierPlan(model_tnet, optimizer_tnet, self.train_dataset, self.test_dataset, paths[1], num_epochs=2)
        trainer.execute()

        model_qnet = PointNetClassifier(40).load_state_dict(model.state_dict(), strict=True)
        qnet = QuaternionNet()
        model_qnet.base.set_input_transform(qnet)
        optimizer_qnet = optim.Adam(qnet.parameters(), lr=1.0E-3)
        trainer = ClassifierPlan(model_qnet, optimizer_qnet, self.train_dataset, self.test_dataset, paths[2], num_epochs=2)
        trainer.execute()

        return model_tnet, model_qnet
