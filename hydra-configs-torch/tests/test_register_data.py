# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_store import ConfigStore
import hydra_configs.torch.utils.data

cs = ConfigStore()


# registers all 'base' optimizer configs with configstore instance
hydra_configs.torch.utils.data.register()

expected = set(
    [
        "dataloader",
        "dataset",
        "chaindataset",
        "concatdataset",
        "iterabledataset",
        "tensordataset",
        "subset",
        "distributedsampler",
        "sampler",
        "batchsampler",
        "randomsampler",
        "sequentialsampler",
        "subsetrandomsampler",
        "weightedrandomsampler",
    ]
)


def test_instantiate_classes() -> None:
    actual = set([conf.split(".yaml")[0] for conf in cs.list("torch/utils/data")])
    assert not actual ^ expected
