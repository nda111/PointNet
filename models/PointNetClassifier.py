import torch.nn as nn
from models.PointNetBase import PointNetBase
from models.modules import PointMLP


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(PointNetClassifier, self).__init__()
        self.num_classes = num_classes

        self.base = PointNetBase()
        self.fc = PointMLP(1024, 512, 256, num_classes, is_conv=False, keep_prob=0.7, use_tail=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.register_module('base', self.base)
        self.register_module('fc', self.fc)
        self.register_module('log_softmax', self.log_softmax)

    def forward(self, x):
        input_transform, feature_transform, _, global_features = self.base(x)
        out = self.fc(global_features)
        out = self.log_softmax(out)

        return input_transform, feature_transform, out
