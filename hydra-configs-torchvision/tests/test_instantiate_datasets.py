# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

import torchvision.datasets as datasets

from typing import Any


@pytest.mark.parametrize(
    "modulepath, classname, cfg, passthrough_args, passthrough_kwargs, expected",
    [
        pytest.param(
            "datasets.vision",
            "VisionDataset",
            {"root": "./dummy_dataset_dir"},
            [],
            {},
            datasets.VisionDataset(root="./dummy_dataset_dir"),
            id="VisionDatasetConf",
        ),
        pytest.param(
            "datasets.mnist",
            "MNIST",
            {"root": "./dummy_data_dir"},
            [],
            {},
            datasets.MNIST(root="./dummy_data_dir"),
            id="MNISTConf",
        ),
        pytest.param(
            "datasets.mnist",
            "FashionMNIST",
            {"root": "./dummy_data_dir"},
            [],
            {},
            datasets.FashionMNIST(root="./dummy_data_dir"),
            id="FashionMNISTConf",
        ),
        pytest.param(
            "datasets.mnist",
            "KMNIST",
            {"root": "./dummy_data_dir"},
            [],
            {},
            datasets.KMNIST(root="./dummy_data_dir"),
            id="KMNISTConf",
        ),
        # TODO: These tests will need to be changed after EMNISTConf and QMNISTConf are corected.
        #        pytest.param(
        #            "datasets.mnist",
        #            "EMNIST",
        #            {"root":'./dummy_data_dir',
        #             "split":"byclass",
        #             "kwargs":None},
        #            [],
        #            {},
        #            datasets.EMNIST(root='./dummy_data_dir', split="byclass"),
        #            id="EMNISTConf",
        #        ),
        #        pytest.param(
        #            "datasets.mnist",
        #            "QMNIST",
        #            {"root":'./dummy_data_dir',
        #             "what":'test',
        #             "compat":None,
        #             "kwargs":None},
        #            [],
        #            {},
        #            datasets.QMNIST('./dummy_data_dir', 'test'),
        #            id="QMNISTConf",
        #        ),
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
