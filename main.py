import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import fmt
from models import PointNetClassifier
from dataset.transforms import FarthestPointSampling, RandomSampling
from dataset import ModelNet40, preprocess_model_net
from experiment import ClassifierTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

raw_path = 'G:\내 드라이브\AIRLAB\Datasets\ModelNet40'
root_path = './ModelNet40_processed'
save_path = './history'

if not os.path.exists(save_path):
    os.mkdir(save_path)
continue_training = False

preprocess_model_net(raw_path, root_path, num_samples=3000)
print()

sampler = RandomSampling(1024)
batch_size = 32
train_dataloader = DataLoader(
    ModelNet40(root_path, sampler=sampler, device=device).train(),
    batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(
    ModelNet40(root_path, sampler=sampler, device=device).test(),
    batch_size=batch_size)

classifier = PointNetClassifier(40).to(device)
optimizer = optim.Adam(params=classifier.parameters(), lr=1.0E-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
start_epoch = 1
train_loss = []
test_loss = []
if continue_training:
    file_name = [name for name in os.listdir(save_path) if name.endswith('.pkl')]
    if len(file_name) != 0:
        file_name.sort(reverse=True)
        file_name = file_name[0]
        bundle = torch.load(os.path.join(save_path, file_name))

        classifier.load_state_dict(bundle['model_state_dict'], strict=True)
        optimizer.load_state_dict(bundle['optim_state_dict'])
        start_epoch = bundle['epochs'] + 1
        train_loss = bundle['train_loss']
        test_loss = bundle['test_loss']

        print('epoch', start_epoch - 1)
        print(f'train_loss={train_loss[-1]}')
        print(f'test_loss={test_loss[-1]}')
        print()

trainer = ClassifierTrainer(train_dataloader, test_dataloader,
                            model=classifier, optimizer=optimizer)
num_epochs = 250
for epoch in range(start_epoch, num_epochs + 1):
    print(f'epoch {epoch}, lr={optimizer.param_groups[0]["lr"]}')

    loss = trainer.train()
    train_loss.append(loss.item())
    scheduler.step()
    print(f'train_loss={loss.item()}')

    loss = trainer.test()
    test_loss.append(loss.item())
    print(f'test_loss={loss.item()}')
    print()

    if epoch % 5 == 0:
        bundle = {
            'epochs': epoch,
            'model_state_dict': classifier.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }

        file_name = f'{fmt.get_timestamp()}.pkl'
        torch.save(bundle, os.path.join(save_path, file_name))
