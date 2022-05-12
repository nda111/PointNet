import torch

from dataset.transforms import Sampling


class FarthestPointSampling(Sampling):
    """
    Sample {num_samples} farthest sample points from point cloud.
    """
    def __init__(self, num_samples: int):
        super(FarthestPointSampling, self).__init__(num_samples)

    @staticmethod
    def sample_points(x: torch.Tensor, num_points: int, ndim: int, num_samples: int) -> torch.Tensor:
        centroid_idx = torch.randint(low=0, high=num_points, size=(1,))
        samples = []
        for _ in range(num_samples):
            centroid = x[centroid_idx].view(1, ndim)
            x = torch.cat([x[:centroid_idx], x[centroid_idx + 1:]], dim=0)

            distances = torch.norm(x - centroid, dim=0)
            farthest_idx = torch.max(distances, dim=0).indices

            samples.append(x[farthest_idx].view(1, ndim))
            centroid_idx = farthest_idx
        samples = torch.cat(samples, dim=0)

        return samples
