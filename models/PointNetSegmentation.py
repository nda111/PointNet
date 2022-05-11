import torch
import torch.nn as nn
from models.PointNetBase import PointNetBase, PointConvChain


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes: int):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes

        self.base = PointNetBase()
        self.mlp = PointConvChain(1088, 512, 256, 128)
        self.out = nn.Conv1d(128, num_classes, kernel_size=1)

        self.register_module('base', self.base)
        self.register_module('mlp', self.mlp)
        self.register_module('out', self.out)

    def forward(self, x):
        input_transform, feature_transform, local_features, global_features = self.base(x)

        global_features = global_features.unsqueeze(2).expand(-1, -1, local_features.size(2))
        features = torch.cat([local_features, global_features], dim=1)

        out = self.mlp(features)
        out = self.out(out).transpose(1, 2).contiguous()

        return input_transform, feature_transform, out
