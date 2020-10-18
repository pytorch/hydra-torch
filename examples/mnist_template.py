from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

###### Hydra Block ######
from typing import List, Any
from omegaconf import MISSING
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

# config schema imports
from config.torch.optim import AdadeltaConf
from config.torch.optim.lr_scheduler import StepLRConf

@dataclass
class CommonArgparseArgs:
    stuff: int = 1
    checkpoint_name: str = 'unnamed.pt'

@dataclass
class ExportedArgparseArgs:
    epochs: int = 14
    batch_size: int = 64
    test_batch_size: int = 1000
    no_cuda: bool = False
    save_model: bool = False
    dry_run: bool = False
    log_interval: int = 10
    seed: int = 1

@dataclass
class MNISTModelConf:
    drop_prob: float = 0.2
    in_features: int = 784
    out_features: int = 10
    hidden_dim: int = 1000

@dataclass
class MNISTConf:
    args: ExportedArgparseArgs = ExportedArgparseArgs()
    model: MNISTModelConf = MNISTModelConf()
    optim: Any = AdadeltaConf()
    scheduler: Any = StepLRConf(step_size=1)


cs = ConfigStore.instance()
cs.store(name="config", node=MNISTConf)

###### / Hydra Block ######

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@hydra.main(config_name='config')
def main(cfg):
    print(cfg.pretty())
    import ipdb; ipdb.set_trace()
    use_cuda = not cfg.args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': cfg.args.batch_size}
    test_kwargs = {'batch_size': cfg.args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #transform=transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))
    #    ])
    #dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                   transform=transform)
    #dataset2 = datasets.MNIST('../data', train=False,
    #                   transform=transform)
    #train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    #test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    for epoch in range(1, cfg.args.epochs + 1):
        train(cfg.args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if cfg.args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
