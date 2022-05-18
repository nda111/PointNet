import random

import torch

from dataset.transforms import Sampling


def triangle_area(pt1, pt2, pt3):
    side_a = torch.norm(pt1 - pt2)
    side_b = torch.norm(pt2 - pt3)
    side_c = torch.norm(pt3 - pt1)
    s = 0.5 * (side_a + side_b + side_c)
    return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5


def sample_point(pt1, pt2, pt3):
    s, t = sorted([random.random(), random.random()])
    f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
    return torch.cat([f(i).view((1, 1)) for i in range(3)], dim=1)


def load_point_cloud(filename: str, sampler: Sampling, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(filename, 'rt') as file:
        line = file.readline().strip()
        if len(line) > 3:
            line = line[3:]
        else:
            line = file.readline().strip()
        num_points, num_faces, _ = [int(val) for val in line.split(' ')]

        points = torch.tensor([[float(val)
                                for val in file.readline().strip().split(' ')]
                               for _ in range(num_points)]).float().to(device)

        faces = [[int(val) for val in file.readline().strip().split(' ')][1:] for _ in range(num_faces)]
        areas = torch.zeros((len(faces))).to(device)
        for i in range(len(areas)):
            areas[i] = triangle_area(points[faces[i][0]],
                                     points[faces[i][1]],
                                     points[faces[i][2]])

        sampled_faces = random.choices(faces, weights=areas, k=int(sampler.num_samples * 1.5))
        pc = []
        for i in range(len(sampled_faces)):
            pc.append(sample_point(points[sampled_faces[i][0]],
                                   points[sampled_faces[i][1]],
                                   points[sampled_faces[i][2]]))

        return sampler(torch.cat(pc, dim=0).transpose(0, 1).contiguous())
