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

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import PointNetClassifier
from experiment.plan import ClassifierPlan

# bundle = torch.load('history/vanilla/20220524_154438.pkl')
# state_dict = bundle['model_state_dict']

model = PointNetClassifier(40).to(device)
# model.base.set_input_transform(nn.Identity().to(device))
# model.load_state_dict(state_dict, strict=True)
optimizer = optim.Adam(model.parameters(), lr=1.0E-3)

plan = ClassifierPlan(model, optimizer, train_dataset, test_dataset, output_path='.vanilla')
report = plan.execute()
torch.save(report, 'report.pkl')  # TODO: 이거 확인하기

# model.eval()
# with torch.no_grad():
#     for sample in DataLoader(test_dataset, batch_size=32):
#         pc = sample['pc']
#         onehot = sample['label']
#         _, _, out = model(pc)
#
#         print(onehot[0])
#         print(out[0])
#
#         input()

