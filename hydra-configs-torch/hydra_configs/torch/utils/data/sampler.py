# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any


@dataclass
class SamplerConf:
    _target_: str = "torch.utils.data.sampler.Sampler"
    data_source: Any = MISSING


@dataclass
class BatchSamplerConf:
    _target_: str = "torch.utils.data.sampler.BatchSampler"
    sampler: Any = MISSING
    batch_size: Any = MISSING
    drop_last: Any = MISSING


@dataclass
class RandomSamplerConf:
    _target_: str = "torch.utils.data.sampler.RandomSampler"
    data_source: Any = MISSING
    replacement: Any = False
    num_samples: Any = None
    generator: Any = None


@dataclass
class SequentialSamplerConf:
    _target_: str = "torch.utils.data.sampler.SequentialSampler"
    data_source: Any = MISSING


@dataclass
class SubsetRandomSamplerConf:
    _target_: str = "torch.utils.data.sampler.SubsetRandomSampler"
    indices: Any = MISSING
    generator: Any = None


@dataclass
class WeightedRandomSamplerConf:
    _target_: str = "torch.utils.data.sampler.WeightedRandomSampler"
    weights: Any = MISSING
    num_samples: Any = MISSING
    replacement: Any = True
    generator: Any = None