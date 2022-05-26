import torch
import torch.nn as nn

from models.modules import PointMaxPool, PointMLP


class TNet(nn.Module):
    def __init__(self, in_dim: int):
        super(TNet, self).__init__()
        self.in_dim = in_dim

        self.mlp = PointMLP(self.in_dim, 64, 128, 1024, is_conv=True, use_tail=True)
        self.pool = PointMaxPool()
        self.fc = PointMLP(1024, 512, 265, self.in_dim * self.in_dim, is_conv=False, use_tail=False)

        self.register_module('mlp', self.mlp)
        self.register_module('pool', self.pool)
        self.register_module('fc', self.fc)

    def forward(self, x):
        t = self.mlp(x)
        t = self.pool(t)

        t = self.fc(t)
        t = t.view(-1, self.in_dim, self.in_dim)
        i = torch.torch.eye(self.in_dim).repeat(t.size(0), 1, 1).to(t.device)
        t = t + i

        out = torch.bmm(t, x)
        return t, out
