import torch
import torch.nn as nn


class PointMaxPool(nn.Module):
    def __init__(self):
        super(PointMaxPool, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=2).values
