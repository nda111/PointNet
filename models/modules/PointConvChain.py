import torch.nn as nn


class PointConvChain(nn.Module):
    def __init__(self, *dims):
        super(PointConvChain, self).__init__()
        self.dims = dims

        self.blocks = []
        for i in range(1, len(dims)):
            in_dim = dims[i - 1]
            out_dim = dims[i]

            block = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )
            self.blocks.append(block)
            self.register_module(f'block{i}', block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x
