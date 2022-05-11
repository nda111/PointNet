import torch.nn as nn
from models.modules import TNet, PointConvChain, PointMaxPool


class PointNetBase(nn.Module):
    def __init__(self):
        super(PointNetBase, self).__init__()

        self.input_transform = TNet(3)
        self.mlp1 = PointConvChain(3, 64, 64)
        self.feature_transform = TNet(64)
        self.mlp2 = PointConvChain(64, 128, 1024)
        self.pool = PointMaxPool()

        self.register_module('input_transform', self.input_transform)
        self.register_module('mlp1', self.mlp1)
        self.register_module('feature_transform', self.feature_transform)
        self.register_module('mlp2', self.mlp2)
        self.register_module('pool', self.pool)

    def forward(self, x):
        input_transform, out = self.input_transform(x)
        out = self.mlp1(out)
        feature_transform, local_features = self.feature_transform(out)
        out = self.mlp2(local_features)
        global_features = self.pool(out)

        return input_transform, feature_transform, local_features, global_features
