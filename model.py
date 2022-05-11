import torch
import torch.nn as nn


class TNet(nn.Module):
    def __init__(self, in_dim: int):
        super(TNet, self).__init__()
