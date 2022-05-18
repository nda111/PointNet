import torch


class RandomJittering:
    def __init__(self, mean: float = 0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch):
        noise = torch.normal(size=x.shape, mean=self.mean, std=self.std).to(x.device)
        return x + noise

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'
