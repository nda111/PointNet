import torch
import torch.nn as nn

from models.modules import PointMaxPool, PointMLP


class QuaternionNet(nn.Module):
    def __init__(self):
        super(QuaternionNet, self).__init__()

        self.mlp = PointMLP(3, 64, 128, 1024, is_conv=True, use_tail=True)
        self.pool = PointMaxPool()
        self.fc = PointMLP(1024, 512, 256, is_conv=False, use_tail=True)
        self.out = nn.Linear(256, 4)

        self.register_module('mlp', self.mlp)
        self.register_module('pool', self.pool)
        self.register_module('fc', self.fc)
        self.register_module('out', self.out)

    def forward(self, x):
        t = self.mlp(x)
        t = self.pool(t)

        t = self.fc(t)
        t = self.out(t)

        t = t.transpose(0, 1).view(4, 1, -1).contiguous()

        qx, qy, qz, qw = t
        qx2, qy2, qz2, qw2 = t ** 2

        t11 = 1 - 2 * (qy2 + qz2).view(-1, 1)
        t22 = 1 - 2 * (qx2 + qz2).view(-1, 1)
        t33 = 1 - 2 * (qx2 + qy2).view(-1, 1)
        t12 = 2 * (qx * qy - qw * qz).view(-1, 1)
        t13 = 2 * (qx * qz - qw * qy).view(-1, 1)
        t23 = 2 * (qy * qz - qw * qx).view(-1, 1)

        t = torch.cat([
            t11, t12, t13,
            t12, t22, t23,
            t13, t23, t33], dim=1).view(-1, 3, 3).contiguous()

        out = torch.matmul(t, x)
        return t, out
