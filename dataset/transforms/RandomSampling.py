import torch

from dataset.transforms import Sampling


class RandomSampling(Sampling):
    """
    Sample {num_samples} random points from point cloud.
    """

    def __init__(self, num_samples: int):
        super(RandomSampling, self).__init__(num_samples)

    @staticmethod
    def sample_points(x: torch.Tensor, num_points: int, ndim: int, num_samples: int) -> torch.Tensor:
        indices = torch.randint(low=0, high=num_points, size=(num_samples,))

        samples = []
        for idx in indices:
            samples.append(x[idx].view(1, ndim))
        samples = torch.cat(samples, dim=0)

        return samples
