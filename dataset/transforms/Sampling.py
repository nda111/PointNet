from abc import ABC, abstractmethod

import torch


class Sampling(ABC):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    @staticmethod
    @abstractmethod
    def sample_points(x: torch.Tensor, num_points: int, ndim: int, num_samples: int) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor):
        x = x.transpose(0, 1)
        num_points = x.size(0)
        ndim = x.size(1)

        if num_points == self.num_samples:
            sample = x
        elif num_points > self.num_samples:
            sample = type(self).sample_points(x.clone(), num_points, ndim, self.num_samples)
        else:
            repeat = self.num_samples // num_points
            num_extra = self.num_samples % num_points
            if num_extra == 0:
                sample = x.repeat(repeat, 1)
            else:
                sample = type(self).sample_points(x.clone(), num_points, ndim, num_extra)
                sample = torch.cat([x.repeat(repeat, 1), sample], dim=0)

        return sample.transpose(0, 1).contiguous()

    def __repr__(self):
        return f'{self.__class__.__name__}(num_samples={self.num_samples})'
