import os

import torch

from experiment.plan import TransformComparisonPlan
from dataset.transforms import RandomSampling
from dataset import ModelNet40, preprocess_model_net

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

raw_path = 'G:\내 드라이브\AIRLAB\Datasets\ModelNet40'
root_path = f'.{os.path.sep}ModelNet40_processed'
save_path = f'.{os.path.sep}history'

if not os.path.exists(save_path):
    os.mkdir(save_path)
continue_training = False

preprocess_model_net(raw_path, root_path, num_samples=3000)
print()

sampler = RandomSampling(1024)
train_dataset = ModelNet40(root_path, sampler=sampler, device=device).train()
test_dataset = ModelNet40(root_path, sampler=sampler, device=device).test()

plan = TransformComparisonPlan(train_dataset, test_dataset, save_path)
report = plan.execute()
torch.save(report, os.path.join(save_path, 'report.pkl'))
