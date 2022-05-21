import torch
import torch.nn as nn

from models.modules import TNet, PointConvChain, PointMaxPool


class PointNetBase(nn.Module):
    def __init__(self, input_transform=TNet(3),
                 feature_transform=TNet(64)):
        super(PointNetBase, self).__init__()

        self.input_transform = input_transform
        if self.input_transform is None:
            self.input_transform = nn.Identity()
        self.mlp1 = PointConvChain(3, 64, 64)
        self.feature_transform = feature_transform
        if self.feature_transform is None:
            self.feature_transform = nn.Identity()
        self.mlp2 = PointConvChain(64, 128, 1024)
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
        output = []

        if type(self.input_transform) is not nn.Identity:
            input_transform, x = self.input_transform(x)
            output.append(input_transform)
        else:
            output.append(torch.eye(3).expand(x.size(0), -1, -1).to(x.device))
        out = self.mlp1(x)
        if type(self.feature_transform) is not nn.Identity:
            feature_transform, out = self.feature_transform(out)
            output.append(feature_transform)
        else:
            output.append(torch.eye(64).expand(x.size(0), -1, -1).to(x.device))
        output.append(out)  # local features
        out = self.mlp2(out)

        global_features = self.pool(out)
        output.append(global_features)

        return tuple(output)
