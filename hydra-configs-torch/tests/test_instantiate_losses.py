# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

import torch.nn.modules.loss as loss

from torch.tensor import Tensor
from typing import Any


@pytest.mark.parametrize(
    "modulepath, classname, cfg, passthrough_args, passthrough_kwargs, expected",
    [
        pytest.param(
            "nn.modules.loss",
            "BCELoss",
            {},
            [],
            {"weight": Tensor([1])},
            loss.BCELoss(),
            id="BCELossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "BCEWithLogitsLoss",
            {},
            [],
            {"weight": Tensor([1]), "pos_weight": Tensor([1])},
            loss.BCEWithLogitsLoss(),
            id="BCEWithLogitsLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "CosineEmbeddingLoss",
            {},
            [],
            {},
            loss.CosineEmbeddingLoss(),
            id="CosineEmbeddingLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "CTCLoss",
            {},
            [],
            {},
            loss.CTCLoss(),
            id="CTCLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "L1Loss",
            {},
            [],
            {},
            loss.L1Loss(),
            id="L1LossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "HingeEmbeddingLoss",
            {},
            [],
            {},
            loss.HingeEmbeddingLoss(),
            id="HingeEmbeddingLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "KLDivLoss",
            {},
            [],
            {},
            loss.KLDivLoss(),
            id="KLDivLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "MarginRankingLoss",
            {},
            [],
            {},
            loss.MarginRankingLoss(),
            id="MarginRankingLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "MSELoss",
            {},
            [],
            {},
            loss.MSELoss(),
            id="MSELossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "MultiLabelMarginLoss",
            {},
            [],
            {},
            loss.MultiLabelMarginLoss(),
            id="MultiLabelMarginLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "MultiLabelSoftMarginLoss",
            {},
            [],
            {"weight": Tensor([1])},
            loss.MultiLabelSoftMarginLoss(),
            id="MultiLabelSoftMarginLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "MultiMarginLoss",
            {},
            [],
            {"weight": Tensor([1])},
            loss.MultiMarginLoss(),
            id="MultiMarginLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "NLLLoss",
            {},
            [],
            {"weight": Tensor([1])},
            loss.NLLLoss(),
            id="NLLLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "NLLLoss2d",
            {},
            [],
            {"weight": Tensor([1])},
            loss.NLLLoss2d(),
            id="NLLLoss2dConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "PoissonNLLLoss",
            {},
            [],
            {},
            loss.PoissonNLLLoss(),
            id="PoissonNLLLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "SmoothL1Loss",
            {},
            [],
            {},
            loss.SmoothL1Loss(),
            id="SmoothL1LossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "SoftMarginLoss",
            {},
            [],
            {},
            loss.SoftMarginLoss(),
            id="SoftMarginLossConf",
        ),
        pytest.param(
            "nn.modules.loss",
            "TripletMarginLoss",
            {},
            [],
            {},
            loss.TripletMarginLoss(),
            id="TripletMarginLossConf",
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
    full_class = f"hydra_configs.torch.{modulepath}.{classname}Conf"
    schema = OmegaConf.structured(get_class(full_class))
    cfg = OmegaConf.merge(schema, cfg)
    obj = instantiate(cfg, *passthrough_args, **passthrough_kwargs)

    assert isinstance(obj, type(expected))
