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
        centroids = torch.zeros((num_samples,)).to(x.device)
        distance = torch.ones((num_points,)).to(x.device) * 1.0E+10
        farthest = torch.randint(low=0, high=num_points, size=(1,)).to(x.device)
        for i in range(num_samples):
            centroids[i] = farthest
            centroid = x[farthest, :]
            dist = torch.sum((x - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, -1)
        samples = x[centroids.long()]

        return samples
