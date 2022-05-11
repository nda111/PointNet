import torch.nn as nn
from models.PointNetBase import PointNetBase


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(PointNetClassifier, self).__init__()
        self.num_classes = num_classes

        self.base = PointNetBase()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.register_module('base', self.base)
        self.register_module('fc', self.fc)

    def forward(self, x):
        input_transform, feature_transform, _, global_features = self.base(x)
        out = self.fc(global_features)

        return input_transform, feature_transform, out
