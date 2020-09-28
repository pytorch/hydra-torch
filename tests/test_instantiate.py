# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

import torch.optim as optim

import torch
from torch import Tensor
from torch import nn
from typing import Any

model = nn.Linear(1, 1)


@pytest.mark.parametrize(
    "classname, cfg, passthrough_kwargs, expected",
    [
        pytest.param(
            "Adadelta",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adadelta(lr=0.1, params=model.parameters()),
            id="AdadeltaConf",
        ),
        pytest.param(
            "Adagrad",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adagrad(lr=0.1, params=model.parameters()),
            id="AdagradConf",
        ),
        pytest.param(
            "Adam",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adam(lr=0.1, params=model.parameters()),
            id="AdamConf",
        ),
        pytest.param(
            "Adamax",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adamax(lr=0.1, params=model.parameters()),
            id="AdamaxConf",
        ),
        pytest.param(
            "AdamW",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.AdamW(lr=0.1, params=model.parameters()),
            id="AdamWConf",
        ),
        pytest.param(
            "ASGD",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.ASGD(lr=0.1, params=model.parameters()),
            id="ASGD",
        ),
        pytest.param(
            "LBFGS",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.LBFGS(lr=0.1, params=model.parameters()),
            id="LBFGS",
        ),
        pytest.param(
            "RMSprop",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.RMSprop(lr=0.1, params=model.parameters()),
            id="RMSprop",
        ),
        pytest.param(
            "Rprop",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Rprop(lr=0.1, params=model.parameters()),
            id="Rprop",
        ),
        pytest.param(
            "SGD",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.SGD(lr=0.1, params=model.parameters()),
            id="SGD",
        ),
        pytest.param(
            "SparseAdam",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.SparseAdam(lr=0.1, params=model.parameters()),
            id="SparseAdam",
        ),
    ],
)
def test_instantiate_classes(
    classname: str, cfg: Any, passthrough_kwargs: Any, expected: Any
) -> None:
    full_class = f"config.torch.optim.{classname}Conf"
    schema = OmegaConf.structured(get_class(full_class))
    cfg = OmegaConf.merge(schema, cfg)
    obj = instantiate(cfg, **passthrough_kwargs)

    def closure():
        return model(Tensor([10]))

    assert torch.all(torch.eq(obj.step(closure), expected.step(closure)))
