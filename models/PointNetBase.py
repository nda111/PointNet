import torch
import torch.nn as nn

from models.modules import TNet, PointMLP, PointMaxPool


class PointNetBase(nn.Module):
    def __init__(self,
                 input_transform=TNet(3),
                 feature_transform=TNet(64)):
        super(PointNetBase, self).__init__()

        self.input_transform = input_transform
        if self.input_transform is None:
            self.input_transform = nn.Identity()
        self.mlp1 = PointMLP(3, 64, 64, is_conv=True, use_tail=True)

        self.feature_transform = feature_transform
        if self.feature_transform is None:
            self.feature_transform = nn.Identity()
        self.mlp2 = PointMLP(64, 128, 1024, is_conv=True, use_tail=True)
        self.pool = PointMaxPool()

        self.register_module('input_transform', self.input_transform)
        self.register_module('mlp1', self.mlp1)
        self.register_module('feature_transform', self.feature_transform)
        self.register_module('mlp2', self.mlp2)
        self.register_module('pool', self.pool)

    def set_input_transform(self, net):
        if net is None:
            net = nn.Identity()
        self.input_transform = net
        self.register_module('input_transform', net)

    def forward(self, x):
        if type(self.input_transform) is not nn.Identity:
            input_transform, x = self.input_transform(x)
        else:
            input_transform = torch.eye(3).repeat(x.size(0), 1).to(x.device)
        point_features = self.mlp1(x)

        if type(self.feature_transform) is not nn.Identity:
            feature_transform, x = self.feature_transform(point_features)
        else:
            feature_transform = torch.eye(64).repeat(x.size(0), 1).to(x.device)
            x = point_features

        global_features = self.pool(x)

        return input_transform, feature_transform, point_features, global_features
