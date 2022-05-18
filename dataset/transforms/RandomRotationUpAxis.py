from typing import Tuple
import torch


class RandomRotationUpAxis:
    def __init__(self, min_angle: float = 0, max_angle: float = torch.pi * 2):
        angles = (min_angle, max_angle)
        self.min = min(angles)
        self.max = max(angles)

    def __call__(self, x: torch):
        theta = torch.rand((1,)) * (self.max - self.min) + self.min
        sin = torch.sin(theta)
        cos = torch.cos(theta)

        mat = torch.tensor([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]]).to(x.device)
        return torch.matmul(mat, x)

    def __repr__(self):
        return f'{self.__class__.__name__}(range={self.min}~{self.max})'
