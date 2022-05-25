import torch.nn as nn


class PointMLP(nn.Module):
    def __init__(self, *dims, is_conv: bool = True, keep_prob: float = 1, use_tail: bool = True):
        super(PointMLP, self).__init__()
        self.dims = dims

        self.blocks = []
        for i in range(1, len(dims)):
            in_dim = dims[i - 1]
            out_dim = dims[i]
            block = [
                nn.Conv1d(in_dim, out_dim, kernel_size=(1,))
                if is_conv else nn.Linear(in_dim, out_dim)
            ]
            if use_tail and i != len(dims) - 1:
                block.extend([
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                ])
                if keep_prob != 1:
                    block.append(nn.Dropout(p=1 - keep_prob))
            block = nn.Sequential(*block)

            self.blocks.append(block)
            self.register_module(f'block{i}', block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
