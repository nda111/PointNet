import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformLoss(nn.Module):
    def __init__(self):
        super(TransformLoss, self).__init__()

    def forward(self, x):
        if x.size(1) != x.size(2):
            raise ValueError

        identity = torch.eye(x.size(1)).expand(x.size(0), -1, -1).to(x.device)
        xt = x.transpose(1, 2).contiguous()
        xxt = torch.matmul(x, xt)

        return F.mse_loss(identity, xxt)
