# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

import torch

# import torchvision.datasets as datasets
import torchvision.transforms as transforms

from typing import Any


def identity(x):
    return x


@pytest.mark.parametrize(
    "modulepath, classname, cfg, passthrough_args, passthrough_kwargs, expected",
    [
        #        pytest.param(
        #            "datasets.vision",
        #            "StandardTransform",
        #            {},
        #            [],
        #            {},
        #            datasets.vision.StandardTransform(),
        #            id="StandardTransformConf",
        #        ),
        pytest.param(
            "transforms.transforms",
            "CenterCrop",
            {"size": (10, 10)},
            [],
            {},
            transforms.transforms.CenterCrop(size=(10, 10)),
            id="CenterCropConf",
        ),
        pytest.param(
            "transforms.transforms",
            "ColorJitter",
            {},
            [],
            {},
            transforms.transforms.ColorJitter(),
            id="ColorJitterConf",
        ),
        pytest.param(
            "transforms.transforms",
            "Compose",
            {"transforms": []},
            [],
            {},
            transforms.transforms.Compose(transforms=[]),
            id="ComposeConf",
        ),
        pytest.param(
            "transforms.transforms",
            "ConvertImageDtype",
            {},
            [],
            {"dtype": torch.int32},
            transforms.transforms.ConvertImageDtype(dtype=torch.int32),
            id="ConvertImageDtypeConf",
        ),
        pytest.param(
            "transforms.transforms",
            "FiveCrop",
            {"size": (10, 10)},
            [],
            {},
            transforms.transforms.FiveCrop(size=(10, 10)),
            id="FiveCropConf",
        ),
        pytest.param(
            "transforms.transforms",
            "Grayscale",
            {},
            [],
            {},
            transforms.transforms.Grayscale(),
            id="GrayscaleConf",
        ),
        pytest.param(
            "transforms.transforms",
            "Lambda",
            {},
            [],
            {"lambd": identity},
            transforms.transforms.Lambda(lambd=identity),
            id="LambdaConf",
        ),
        pytest.param(
            "transforms.transforms",
            "LinearTransformation",
            {},
            [],
            {
                "transformation_matrix": torch.eye(2),
                "mean_vector": torch.Tensor([1, 1]),
            },
            transforms.transforms.LinearTransformation(
                transformation_matrix=torch.eye(2), mean_vector=torch.Tensor([1, 1])
            ),
            id="LinearTransformationConf",
        ),
        pytest.param(
            "transforms.transforms",
            "Normalize",
            {"mean": 0, "std": 1},
            [],
            {},
            transforms.transforms.Normalize(mean=0, std=1),
            id="NormalizeConf",
        ),
        pytest.param(
            "transforms.transforms",
            "Pad",
            {"padding": 0},
            [],
            {},
            transforms.transforms.Pad(padding=0),
            id="PaddingConf",
        ),
        pytest.param(
            "transforms.transforms",
            "PILToTensor",
            {},
            [],
            {},
            transforms.transforms.PILToTensor(),
            id="PILToTensorConf",
        ),
        pytest.param(
            "transforms.transforms",
            "RandomAffine",
            {"degrees": 0},
            [],
            {},
            transforms.transforms.RandomAffine(degrees=0),
            id="RandomAffineConf",
        ),
        # pytest.param(
        #    "transforms.transforms",
        #    "RandomApply",
        #    {},
        #    [[ToTensor()]],
        #    {},
        #    transforms.transforms.RandomApply([ToTensor()]),
        #    id="RandomApplyConf",
        # ),
        # pytest.param(
        #    "transforms.transforms",
        #    "RandomChoice",
        #    {},
        #    [],
        #    {"transforms":[[ToTensor()]]},
        #    transforms.transforms.RandomChoice([ToTensor()]),
        #    id="RandomChoiceConf",
        # ),
        pytest.param(
            "transforms.transforms",
            "RandomCrop",
            {"size": (10, 10)},
            [],
            {},
            transforms.transforms.RandomCrop(size=(10, 10)),
            id="RandomCropConf",
        ),
        pytest.param(
            "transforms.transforms",
            "RandomErasing",
            {},
            [],
            {},
            transforms.transforms.RandomErasing(),
            id="RandomErasingConf",
        ),
        pytest.param(
            "transforms.transforms",
            "RandomGrayscale",
            {},
            [],
            {},
            transforms.transforms.RandomGrayscale(),
            id="RandomGrayscaleConf",
        ),
        pytest.param(
            "transforms.transforms",
            "RandomHorizontalFlip",
            {},
            [],
            {},
            transforms.transforms.RandomHorizontalFlip(),
            id="RandomHorizontalFlipConf",
        ),
        # pytest.param(
        #     "transforms.transforms",
        #     "RandomOrder",
        #     {},
        #     [],
        #     {},
        #     transforms.transforms.RandomOrder(),
        #     id="RandomOrderConf",
        # ),
        pytest.param(
            "transforms.transforms",
            "RandomPerspective",
            {},
            [],
            {},
            transforms.transforms.RandomPerspective(),
            id="RandomPerspectiveConf",
        ),
        pytest.param(
            "transforms.transforms",
            "RandomResizedCrop",
            {"size": (10, 10)},
            [],
            {},
            transforms.transforms.RandomResizedCrop(size=(10, 10)),
            id="RandomResizedCropConf",
        ),
        pytest.param(
            "transforms.transforms",
            "RandomRotation",
            {"degrees": 0},
            [],
            {},
            transforms.transforms.RandomRotation(degrees=0),
            id="RandomRotationConf",
        ),
        # pytest.param(
        #    "transforms.transforms",
        #    "RandomTransforms",
        #    {},
        #    [],
        #    {},
        #    transforms.transforms.RandomTransforms(),
        #    id="RandomTransformsConf",
        # ),
        pytest.param(
            "transforms.transforms",
            "RandomVerticalFlip",
            {},
            [],
            {},
            transforms.transforms.RandomVerticalFlip(),
            id="RandomVerticalFlipConf",
        ),
        pytest.param(
            "transforms.transforms",
            "Resize",
            {"size": (10, 10)},
            [],
            {},
            transforms.transforms.Resize(size=(10, 10)),
            id="ResizeConf",
        ),
        pytest.param(
            "transforms.transforms",
            "TenCrop",
            {"size": (10, 10)},
            [],
            {},
            transforms.transforms.TenCrop(size=(10, 10)),
            id="TenCropConf",
        ),
        pytest.param(
            "transforms.transforms",
            "ToPILImage",
            {},
            [],
            {},
            transforms.transforms.ToPILImage(),
            id="ToPILImageConf",
        ),
        pytest.param(
            "transforms.transforms",
            "ToTensor",
            {},
            [],
            {},
            transforms.transforms.ToTensor(),
            id="ToTensorConf",
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
