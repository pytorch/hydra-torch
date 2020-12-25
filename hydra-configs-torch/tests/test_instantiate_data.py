# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

import torch.utils.data as data

import torch
from typing import Any

dummy_tensor = torch.tensor((1, 1))
dummy_dataset = data.dataset.TensorDataset(dummy_tensor)
dummy_sampler = data.Sampler(data_source=dummy_dataset)


@pytest.mark.parametrize(
    "modulepath, classname, cfg, passthrough_args, passthrough_kwargs, expected",
    [
        pytest.param(
            "utils.data.dataloader",
            "DataLoader",
            {"batch_size": 4},
            [],
            {"dataset": dummy_dataset},
            data.DataLoader(batch_size=4, dataset=dummy_dataset),
            id="DataLoaderConf",
        ),
        pytest.param(
            "utils.data.dataset",
            "Dataset",
            {},
            [],
            {},
            data.Dataset(),
            id="DatasetConf",
        ),
        pytest.param(
            "utils.data.dataset",
            "ChainDataset",
            {},
            [],
            {"datasets": [dummy_dataset, dummy_dataset]},
            data.ChainDataset(datasets=[dummy_dataset, dummy_dataset]),
            id="ChainDatasetConf",
        ),
        pytest.param(
            "utils.data.dataset",
            "ConcatDataset",
            {},
            [],
            {"datasets": [dummy_dataset, dummy_dataset]},
            data.ConcatDataset(datasets=[dummy_dataset, dummy_dataset]),
            id="ConcatDatasetConf",
        ),
        pytest.param(
            "utils.data.dataset",
            "IterableDataset",
            {},
            [],
            {},
            data.IterableDataset(),
            id="IterableDatasetConf",
        ),
        # TODO: investigate asterisk in signature instantiation limitation
        # pytest.param(
        #    "utils.data.dataset",
        #    "TensorDataset",
        #    {},
        #    [],
        #    {"tensors":[dummy_tensor]},
        #    data.TensorDataset(dummy_tensor),
        #    id="TensorDatasetConf",
        # ),
        pytest.param(
            "utils.data.dataset",
            "Subset",
            {},
            [],
            {"dataset": dummy_dataset, "indices": [0]},
            data.Subset(dummy_dataset, 0),
            id="SubsetConf",
        ),
        pytest.param(
            "utils.data.sampler",
            "Sampler",
            {},
            [],
            {"data_source": dummy_dataset},
            data.Sampler(data_source=dummy_dataset),
            id="SamplerConf",
        ),
        pytest.param(
            "utils.data.sampler",
            "BatchSampler",
            {"batch_size": 4, "drop_last": False},
            [],
            {"sampler": dummy_sampler},
            data.BatchSampler(sampler=dummy_sampler, batch_size=4, drop_last=False),
            id="BatchSamplerConf",
        ),
        pytest.param(
            "utils.data.sampler",
            "RandomSampler",
            {},
            [],
            {"data_source": dummy_dataset},
            data.RandomSampler(data_source=dummy_dataset),
            id="RandomSamplerConf",
        ),
        pytest.param(
            "utils.data.sampler",
            "SequentialSampler",
            {},
            [],
            {"data_source": dummy_dataset},
            data.SequentialSampler(data_source=dummy_dataset),
            id="SequentialSamplerConf",
        ),
        pytest.param(
            "utils.data.sampler",
            "SubsetRandomSampler",
            {"indices": [1]},
            [],
            {},
            data.SubsetRandomSampler(indices=[1]),
            id="SubsetRandomSamplerConf",
        ),
        pytest.param(
            "utils.data.sampler",
            "WeightedRandomSampler",
            {"weights": [1], "num_samples": 1},
            [],
            {},
            data.WeightedRandomSampler(weights=[1], num_samples=1),
            id="WeightedRandomSamplerConf",
        ),
        # TODO: investigate testing distributed instantiation
        # pytest.param(
        #    "utils.data.distributed",
        #    "DistributedSampler",
        #    {},
        #    [],
        #    {"dataset": dummy_dataset},
        #    data.DistributedSampler(group=dummy_group,dataset=dummy_dataset),
        #    id="DistributedSamplerConf",
        # ),
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
