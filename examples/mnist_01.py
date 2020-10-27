# flake8: noqa
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR

###### HYDRA BLOCK ######
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

# hydra-torch structured config imports
from config.torch.optim import AdadeltaConf
from config.torch.optim.lr_scheduler import StepLRConf


@dataclass
class MNISTConf:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 14
    no_cuda: bool = False
    dry_run: bool = False
    seed: int = 1
    log_interval: int = 10
    save_model: bool = False
    checkpoint_name: str = "unnamed.pt"
    model: Any = MNISTNetConf()
    optim: Any = AdadeltaConf()
    scheduler: Any = StepLRConf(
        step_size=1
    )  # we pass a default for step_size since it is required, but missing a default in PyTorch (and consequently in hydra-torch)


@dataclass
class MNISTNetConf:
    conv1_out_channels: int = 32
    conv2_out_channels: int = 64
    maxpool1_kernel: int = 2
    dropout1_prob: float = 0.25
    dropout2_prob: float = 0.5
    fc_hidden_features: int = 128


cs = ConfigStore.instance()
cs.store(name="mnistconf", node=MNISTConf)

###### / HYDRA BLOCK ######


class Net(nn.Module):
    def __init__(self, input_shape, output_shape, cfg):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, cfg.model.conv1_out_channels, 3, 1)
        self.conv2 = nn.Conv2d(
            cfg.model.conv1_out_channels, cfg.model.conv2_out_channels, 3, 1
        )
        self.dropout1 = nn.Dropout2d(cfg.model.dropout1_prob)
        self.dropout2 = nn.Dropout2d(cfg.model.dropout2_prob)

        conv_out_shape = self._compute_conv_out_shape(input_shape)
        linear_in_shape = torch.prod(conv_out_shape)

        self.fc1 = nn.Linear(linear_in_shape, cfg.model.fc_hidden_features)
        self.fc2 = nn.Linear(cfg.model.fc_hidden_features, torch.prod(output_shape))

    def _compute_conv_out_shape(self, input_shape):
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            dummy_output = F.max_pool2d(x, cfg.model.maxpool1_kernel)
        return dummy_output.shape()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, cfg.model.maxpool1_kernel)
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


@hydra.main(config_name="mnistconf")
def main(cfg):  # DIFF
    print(cfg.pretty())
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()  # DIFF
    torch.manual_seed(cfg.seed)  # DIFF
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": cfg.batch_size}  # DIFF
    test_kwargs = {"batch_size": cfg.test_batch_size}  # DIFF
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    optimizer = Adadelta(
        lr=cfg.adadelta.lr,
        rho=cfg.adadelta.rho,
        eps=cfg.adadelta.eps,
        weight_decay=cfg.adadelta.weight_decay,
        params=model.parameters(),
    )  # DIFF
    scheduler = StepLR(
        step_size=cfg.steplr.step_size,
        gamma=cfg.steplr.gamma,
        last_epoch=cfg.steplr.last_epoch,
        optimizer=optimizer,
    )  # DIFF

    for epoch in range(1, cfg.epochs + 1):  # DIFF
        train(cfg, model, device, train_loader, optimizer, epoch)  # DIFF
        test(model, device, test_loader)
        scheduler.step()

    if cfg.save_model:  # DIFF
        torch.save(model.state_dict(), cfg.checkpoint_name)  # DIFF


if __name__ == "__main__":
    main()
