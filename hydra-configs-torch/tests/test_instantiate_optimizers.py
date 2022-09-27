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
    "modulepath, classname, cfg, passthrough_kwargs, expected",
    [
        pytest.param(
            "optim.adadelta",
            "Adadelta",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adadelta(lr=0.1, params=model.parameters()),
            id="AdadeltaConf",
        ),
        pytest.param(
            "optim.adagrad",
            "Adagrad",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adagrad(lr=0.1, params=model.parameters()),
            id="AdagradConf",
        ),
        pytest.param(
            "optim.adam",
            "Adam",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adam(lr=0.1, params=model.parameters()),
            id="AdamConf",
        ),
        pytest.param(
            "optim.adamax",
            "Adamax",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Adamax(lr=0.1, params=model.parameters()),
            id="AdamaxConf",
        ),
        pytest.param(
            "optim.adamw",
            "AdamW",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.AdamW(lr=0.1, params=model.parameters()),
            id="AdamWConf",
        ),
        pytest.param(
            "optim.asgd",
            "ASGD",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.ASGD(lr=0.1, params=model.parameters()),
            id="ASGDConf",
        ),
        pytest.param(
            "optim.lbfgs",
            "LBFGS",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.LBFGS(lr=0.1, params=model.parameters()),
            id="LBFGSConf",
        ),
        pytest.param(
            "optim.rmsprop",
            "RMSprop",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.RMSprop(lr=0.1, params=model.parameters()),
            id="RMSpropConf",
        ),
        pytest.param(
            "optim.rprop",
            "Rprop",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.Rprop(lr=0.1, params=model.parameters()),
            id="RpropConf",
        ),
        pytest.param(
            "optim.sgd",
            "SGD",
            {"lr": 0.1},
            {"params": model.parameters()},
            optim.SGD(lr=0.1, params=model.parameters()),
            id="SGDConf",
        ),
        pytest.param(
            "optim.sparse_adam",
            "SparseAdam",
            {"lr": 0.1},
            {"params": list(model.parameters())},
            optim.SparseAdam(lr=0.1, params=list(model.parameters())),
            id="SparseAdamConf",
        ),
    ],
)
def test_instantiate_classes(
    modulepath: str, classname: str, cfg: Any, passthrough_kwargs: Any, expected: Any
) -> None:
    full_class = f"hydra_configs.torch.{modulepath}.{classname}Conf"
    schema = OmegaConf.structured(get_class(full_class))
    cfg = OmegaConf.merge(schema, cfg)
    obj = instantiate(cfg, **passthrough_kwargs)

    def closure():
        return model(Tensor([10]))

    assert torch.all(torch.eq(obj.step(closure), expected.step(closure)))
