import os
import random
from tqdm.auto import tqdm

import torch

from utils import get_device


def triangle_area(pt1, pt2, pt3):
    points = torch.cat([pt1, pt2, pt3], dim=0).float()
    sides = torch.cat([
        points[0] - points[1],
        points[0] - points[2],
        points[1] - points[2],
    ], dim=0).view(3, 3)
    sides = torch.norm(sides, dim=1)

    s = 0.5 * torch.sum(sides)
    sub = s - sides
    return max(s * sub[0] * sub[1] * sub[2], 0) ** 0.5


def sample_point(pt1, pt2, pt3):
    points = torch.cat([pt1, pt2, pt3], dim=0).float()
    s, t = torch.sort(torch.rand(2)).values
    w = torch.tensor([s, t - s, 1 - t])
    return torch.sum(points.transpose(0, 1) * w, dim=1).view(1, -1)


def load_point_cloud(input_filename: str, output_filename: str, num_samples: int, device: str = get_device()):
    with open(input_filename, 'rt') as file:
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

        sampled_faces = random.choices(faces, weights=areas, k=num_samples)
        pc = []
        for i in range(len(sampled_faces)):
            pc.append(sample_point(points[sampled_faces[i][0]],
                                   points[sampled_faces[i][1]],
                                   points[sampled_faces[i][2]]))

    pc = torch.cat(pc, dim=0).transpose(0, 1).contiguous()
    torch.save(pc, output_filename)
    return pc


def preprocess_model_net(root_path: str, output_path: str, num_samples: int = 2048,
                         overwrite: bool = False, verbose: bool = True, device: str = get_device()):
    def _print(*args, sep=' ', end='\n'):
        if verbose:
            print(*args, sep=sep, end=end)

    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            _print('ModelNet40 was already preprocessed.')
            return

    os.mkdir(output_path)
    classes = os.listdir(root_path)
    for cls_idx, cls in enumerate(classes):
        _print(cls_idx + 1, '/', len(classes), '\t:', cls)

        os.mkdir(os.path.join(output_path, cls))
        for mode in ('train', 'test'):
            _print(mode)
            os.mkdir(os.path.join(output_path, cls, mode))
            dir_path = os.path.join(root_path, cls, mode)

            def process_file(fn: str):
                fullname = os.path.join(dir_path, fn)
                output_filename = os.path.join(output_path, cls, mode, fn)
                load_point_cloud(fullname, output_filename, num_samples=num_samples, device=device)

            if verbose:
                for filename in tqdm(os.listdir(dir_path)):
                    process_file(filename)
            else:
                for filename in os.listdir(dir_path):
                    process_file(filename)
