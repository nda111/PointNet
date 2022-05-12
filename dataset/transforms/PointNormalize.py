import torch


class PointNormalize:
    def __call__(self, x: torch.Tensor):
        # centered = x - torch.mean(x, dim=0)
        # out = centered / torch.max(torch.norm(centered, dim=1))
        # return out

        # centered = x - torch.mean(x, dim=0)
        # return centered / torch.max(centered)
    
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'
