import os

import torch

import utils.format
import utils.format as fmt


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def clone_object(obj):
    filename = f'{utils.format.get_timestamp()}.tmp'
    torch.save(obj, filename)

    new = torch.load(obj)
    os.remove(filename)
    return new
