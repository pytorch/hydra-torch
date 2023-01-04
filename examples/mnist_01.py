# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

###### HYDRA BLOCK ###### # noqa: E266
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from typing import Any
from dataclasses import dataclass

# hydra-torch structured config imports:
from hydra_configs.torch.optim import AdadeltaConf
from hydra_configs.torch.optim.lr_scheduler import StepLRConf
from hydra_configs.torch.utils.data.dataloader import DataLoaderConf

from hydra_configs.torchvision.datasets.mnist import MNISTConf

# NOTE:^ Above still uses hydra_configs namespace, but comes from .torchvision package


@dataclass
class MNISTNetConf:
    conv1_out_channels: int = 32
    conv2_out_channels: int = 64
    maxpool1_kernel_size: int = 2
    dropout1_prob: float = 0.25
    dropout2_prob: float = 0.5
    fc_hidden_features: int = 128


@dataclass
class TopLvlConf:
    epochs: int = 14
    no_cuda: bool = False
    dry_run: bool = False
    seed: int = 1
    log_interval: int = 10
    save_model: bool = False
    checkpoint_name: str = "unnamed.pt"
    train_dataloader: DataLoaderConf = DataLoaderConf(
        batch_size=64, shuffle=True, num_workers=1, pin_memory=False
    )
    test_dataloader: DataLoaderConf = DataLoaderConf(
        batch_size=1000, shuffle=False, num_workers=1
    )
    train_dataset: MNISTConf = MNISTConf(root="../data", train=True, download=True)
    test_dataset: MNISTConf = MNISTConf(root="../data", train=False, download=True)
    model: MNISTNetConf = MNISTNetConf()
    optim: Any = AdadeltaConf()
    scheduler: Any = StepLRConf(step_size=1)


cs = ConfigStore.instance()
cs.store(name="toplvlconf", node=TopLvlConf)

###### / HYDRA BLOCK ###### # noqa: E266


class Net(nn.Module):
    # DIFF: new model definition with configurable params
    def __init__(self, input_shape, output_shape, cfg):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, cfg.model.conv1_out_channels, 3, 1)
        self.conv2 = nn.Conv2d(
            cfg.model.conv1_out_channels, cfg.model.conv2_out_channels, 3, 1
        )
        self.dropout1 = nn.Dropout2d(cfg.model.dropout1_prob)
        self.dropout2 = nn.Dropout2d(cfg.model.dropout2_prob)
        self.maxpool1 = nn.MaxPool2d(cfg.model.maxpool1_kernel_size)

        conv_out_shape = self._compute_conv_out_shape(input_shape)
        linear_in_shape = conv_out_shape.numel()

        self.fc1 = nn.Linear(linear_in_shape, cfg.model.fc_hidden_features)
        self.fc2 = nn.Linear(cfg.model.fc_hidden_features, output_shape[1])

    # /DIFF

    # DIFF: new utility method (incidental, not critical)
    def _compute_conv_out_shape(self, input_shape):
        dummy_input = torch.zeros(input_shape).unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            dummy_output = self.maxpool1(x)
        return dummy_output.shape

    # /DIFF

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)  # DIFF
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
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
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
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


@hydra.main(config_name="toplvlconf")
def main(cfg):
    print(cfg.pretty())
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # DIFF: the following are removed as they are now in DataloaderConf
    # train_kwargs = {"batch_size": cfg.batch_size}
    # test_kwargs = {"batch_size": cfg.test_batch_size}
    # if use_cuda:
    #    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    #    train_kwargs.update(cuda_kwargs)
    #    test_kwargs.update(cuda_kwargs)
    # /DIFF

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # DIFF: hotswap enabled datasets with fixed transforms
    train_dataset = instantiate(cfg.train_dataset, transform=transform)
    test_dataset = instantiate(cfg.test_dataset, transform=transform)
    train_loader = instantiate(cfg.train_dataloader, dataset=train_dataset)
    test_loader = instantiate(cfg.test_dataloader, dataset=test_dataset)
    # /DIFF

    # DIFF: explicit I/O, configurable model
    input_shape = (1, 28, 28)
    output_shape = (1, 10)
    model = Net(input_shape, output_shape, cfg).to(device)
    # /DIFF

    # DIFF: hotswap enabled optimizer/scheduler
    optimizer = instantiate(cfg.optim, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    # /DIFF

    for epoch in range(1, cfg.epochs + 1):
        train(cfg, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if cfg.save_model:
        torch.save(model.state_dict(), cfg.checkpoint_name)


if __name__ == "__main__":
    main()
