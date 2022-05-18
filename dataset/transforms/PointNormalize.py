import torch


class PointNormalize:
    def __call__(self, x: torch.Tensor):
        centroid = torch.mean(x, dim=1).view(-1, 1)
        x = x - centroid
        distance = torch.max(torch.sqrt(torch.sum(x ** 2, dim=0)), dim=0).values
        x = x / distance
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'
