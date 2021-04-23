# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# flake8: noqa
from .dataloader import DataLoaderConf
from .dataset import DatasetConf
from .dataset import ChainDatasetConf
from .dataset import ConcatDatasetConf
from .dataset import IterableDatasetConf
from .dataset import TensorDatasetConf
from .dataset import SubsetConf
from .distributed import DistributedSamplerConf
from .sampler import SamplerConf
from .sampler import BatchSamplerConf
from .sampler import RandomSamplerConf
from .sampler import SequentialSamplerConf
from .sampler import SubsetRandomSamplerConf
from .sampler import WeightedRandomSamplerConf
from hydra.core.config_store import ConfigStoreWithProvider


def register():
    with ConfigStoreWithProvider("torch") as cs:
        cs.store(group="torch/utils/data", name="dataloader", node=DataLoaderConf)
        cs.store(group="torch/utils/data", name="dataset", node=DatasetConf)
        cs.store(group="torch/utils/data", name="chaindataset", node=ChainDatasetConf)
        cs.store(group="torch/utils/data", name="concatdataset", node=ConcatDatasetConf)
        cs.store(
            group="torch/utils/data", name="iterabledataset", node=IterableDatasetConf
        )
        cs.store(group="torch/utils/data", name="tensordataset", node=TensorDatasetConf)
        cs.store(group="torch/utils/data", name="subset", node=SubsetConf)
        cs.store(
            group="torch/utils/data",
            name="distributedsampler",
            node=DistributedSamplerConf,
        )
        cs.store(group="torch/utils/data", name="sampler", node=SamplerConf)
        cs.store(group="torch/utils/data", name="batchsampler", node=BatchSamplerConf)
        cs.store(group="torch/utils/data", name="randomsampler", node=RandomSamplerConf)
        cs.store(
            group="torch/utils/data",
            name="sequentialsampler",
            node=SequentialSamplerConf,
        )
        cs.store(
            group="torch/utils/data",
            name="subsetrandomsampler",
            node=SubsetRandomSamplerConf,
        )
        cs.store(
            group="torch/utils/data",
            name="weightedrandomsampler",
            node=WeightedRandomSamplerConf,
        )
