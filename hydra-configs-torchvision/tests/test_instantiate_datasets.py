# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pytest
from pathlib import Path
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf
from typing import Any

import torch
import torchvision.datasets as datasets


@pytest.mark.parametrize(
    "modulepath, classname, cfg, passthrough_args, passthrough_kwargs, expected_class",
    [
        pytest.param(
            "datasets.vision",
            "VisionDataset",
            {"root": None},
            [],
            {},
            datasets.VisionDataset,
            id="VisionDatasetConf",
        ),
        pytest.param(
            "datasets.mnist",
            "MNIST",
            {"root": None},
            [],
            {},
            datasets.MNIST,
            id="MNISTConf",
        ),
        pytest.param(
            "datasets.mnist",
            "FashionMNIST",
            {"root": None},
            [],
            {},
            datasets.FashionMNIST,
            id="FashionMNISTConf",
        ),
        pytest.param(
            "datasets.mnist",
            "KMNIST",
            {"root": None},
            [],
            {},
            datasets.KMNIST,
            id="KMNISTConf",
        ),
        # TODO: These tests will need to be changed after blockers:
        # 1. EMNISTConf and QMNISTConf are manually created
        # 2. hydra.utils.instantiate is updated to allow *kwargs instantiation
        #        pytest.param(
        #            "datasets.mnist",
        #            "EMNIST",
        #            {"root":None,
        #             "split":"byclass",
        #             "kwargs":None},
        #            [],
        #            {},
        #            datasets.EMNIST,
        #            id="EMNISTConf",
        #        ),
        #        pytest.param(
        #            "datasets.mnist",
        #            "QMNIST",
        #            {"root":None,
        #             "what":'test',
        #             "compat":None,
        #             "kwargs":None},
        #            [],
        #            {},
        #            datasets.QMNIST,
        #            id="QMNISTConf",
        #        ),
    ],
)
def test_instantiate_classes(
    tmpdir: Path,
    modulepath: str,
    classname: str,
    cfg: Any,
    passthrough_args: Any,
    passthrough_kwargs: Any,
    expected_class: Any,
) -> None:

    # Create fake dataset and put it in tmpdir for test:
    tmp_data_root = tmpdir.mkdir("data")
    processed_dir = os.path.join(tmp_data_root, classname, "processed")
    os.makedirs(processed_dir)
    torch.save(torch.tensor([[1.0], [1.0]]), processed_dir + "/training.pt")
    torch.save(torch.tensor([1.0]), processed_dir + "/test.pt")

    # cfg is populated here since it requires tmpdir testfixture
    cfg["root"] = str(tmp_data_root)
    full_class = f"hydra_configs.torchvision.{modulepath}.{classname}Conf"
    schema = OmegaConf.structured(get_class(full_class))
    cfg = OmegaConf.merge(schema, cfg)
    obj = instantiate(cfg, *passthrough_args, **passthrough_kwargs)
    expected_obj = expected_class(root=tmp_data_root)

    assert isinstance(obj, type(expected_obj))
