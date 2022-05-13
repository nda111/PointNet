import torch


class PointNormalize:
    def __call__(self, x: torch.Tensor):
        _min = torch.min(x, dim=1).values.view(-1, 1)
        _max = torch.max(x, dim=1).values.view(-1, 1)
        scale = _max - _min

        out = (x - _min) * 2 / scale - 1
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}()'
