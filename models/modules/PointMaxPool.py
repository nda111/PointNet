import torch
import torch.nn as nn


class PointMaxPool(nn.Module):
    def __init__(self):
        super(PointMaxPool, self).__init__()

    def forward(self, x):
        m1 = torch.max(x, dim=2, keepdim=True)[0].view(x.size(0), -1)
        return m1
