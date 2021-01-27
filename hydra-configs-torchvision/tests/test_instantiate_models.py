# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

import torchvision.models as models


from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from typing import Any

bb = BasicBlock(10, 10)
mnasnet_dict = {"alpha": 1.0, "num_classes": 1000}


@pytest.mark.parametrize(
    "modulepath, classname, cfg, passthrough_args, passthrough_kwargs, expected",
    [
        pytest.param(
            "models.alexnet",
            "AlexNet",
            {},
            [],
            {},
            models.AlexNet(),
            id="AlexNetConf",
        ),
        pytest.param(
            "models.resnet",
            "ResNet",
            {"layers": [2, 2, 2, 2]},
            [],
            {"block": Bottleneck},
            models.ResNet(block=Bottleneck, layers=[2, 2, 2, 2]),
            id="ResNetConf",
        ),
        pytest.param(
            "models.densenet",
            "DenseNet",
            {},
            [],
            {},
            models.DenseNet(),
            id="DenseNetConf",
        ),
        pytest.param(
            "models.squeezenet",
            "SqueezeNet",
            {},
            [],
            {},
            models.SqueezeNet(),
            id="SqueezeNetConf",
        ),
        pytest.param(
            "models.mnasnet",
            "MNASNet",
            {"alpha": 1.0},
            [],
            {},
            models.MNASNet(alpha=1.0),
            id="MNASNetConf",
        ),
        pytest.param(
            "models.googlenet",
            "GoogLeNet",
            {},
            [],
            {},
            models.GoogLeNet(),
            id="GoogleNetConf",
        ),
    ],
)
def test_instantiate_classes(
    modulepath: str,
    classname: str,
    cfg: Any,
    passthrough_args: Any,
    passthrough_kwargs: Any,
    expected: Any,
) -> None:
    full_class = f"hydra_configs.torchvision.{modulepath}.{classname}Conf"
    schema = OmegaConf.structured(get_class(full_class))
    cfg = OmegaConf.merge(schema, cfg)
    obj = instantiate(cfg, *passthrough_args, **passthrough_kwargs)

    assert isinstance(obj, type(expected))
