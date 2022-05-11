import torch
import torch.nn as nn

from models.modules import PointMaxPool, PointConvChain


class TNet(nn.Module):
    def __init__(self, in_dim: int):
        super(TNet, self).__init__()
        self.in_dim = in_dim

        def fc_block(in_dim: int, out_dim: int):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )

        self.mlp = PointConvChain(self.in_dim, 64, 128, 1024)
        self.pool = PointMaxPool()
        self.fc = nn.Sequential(
            fc_block(1024, 512),
            fc_block(512, 256),
            fc_block(256, 512),
            fc_block(512, 256)
        )
        self.out = nn.Linear(256, self.in_dim * self.in_dim)

        self.register_module('mlp', self.mlp)
        self.register_module('pool', self.pool)
        self.register_module('fc', self.fc)
        self.register_module('out', self.out)

    def forward(self, x):
        t = self.mlp(x)
        t = self.pool(t)

        t = self.fc(t)
        t = self.out(t)
        t = t.reshape(-1, self.in_dim, self.in_dim)

        out = torch.matmul(t, x)
        return t, out
