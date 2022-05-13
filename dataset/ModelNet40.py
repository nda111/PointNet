import os
from typing import Any
import math

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from dataset.transforms import PointNormalize, Sampling


class ModelNet40(data.Dataset):
    def __init__(self, root_path: str, sampler: Sampling, transform: Any = None, device=None, limit: int = -1):
        super(ModelNet40, self).__init__()

        self.root_path = root_path
        self.num_samples = sampler.num_samples
        self.transform = transform
        self.basic_transform = transforms.Compose([
            sampler,
            PointNormalize(),
        ])
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        if limit <= 0:
            raise ValueError
        self.limit = limit

        self.mode = 'train'
        self.class_files = []
        self.label2id = {}
        self.id2label = {}
        self.load_metadata()

    def train(self):
        self.mode = 'train'
        self.load_metadata()
        return self

    def test(self):
        self.mode = 'test'
        self.load_metadata()
        return self

    def load_metadata(self):
        self.class_files.clear()
        classes = os.listdir(self.root_path)
        for idx, cls in enumerate(classes):
            # Vector encoding
            self.label2id[cls] = idx
            self.id2label[idx] = cls

            # Path
            path = os.path.join(self.root_path, cls, self.mode)
            file_names = os.listdir(path)
            for file_name in file_names:
                self.class_files.append((cls, file_name))

    def __len__(self):
        if self.limit == -1:
            return len(self.class_files)
        else:
            return self.limit

    def __getitem__(self, idx):
        if self.limit == -1:
            idx = math.floor((len(self.class_files) - 1) / (self.limit - 1) * idx)

        cls, file_name = self.class_files[idx]
        fullname = os.path.join(self.root_path, cls, self.mode, file_name)

        points = []
        with open(fullname, 'rt') as file:
            line = file.readline().upper()[:-1]
            if line == 'OFF':
                line = file.readline()
            elif line.upper().startswith('OFF'):
                line = line[3:]
            else:
                raise ValueError

            num_points = int(line.split(' ')[0])
            for _ in range(num_points):
                line = file.readline().split(' ')[:3]
                point = list(map(float, line))
                points.append(point)
        pc = torch.tensor(points).to(self.device)

        pc = pc.transpose(0, 1).contiguous()
        pc = self.basic_transform(pc)
        if self.transform is not None:
            pc = self.transform(pc)

        onehot = torch.zeros((len(self.id2label),), dtype=torch.float).to(self.device)
        onehot[self.label2id[cls]] = 1
        return {'label': onehot, 'pc': pc}

