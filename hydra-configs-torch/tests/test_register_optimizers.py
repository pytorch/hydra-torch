# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_store import ConfigStore
import hydra_configs.torch.optim

cs = ConfigStore()

# registers all 'base' optimizer configs with configstore instance
hydra_configs.torch.optim.register()

expected = set(
    [
        "adadelta",
        "adagrad",
        "adam",
        "adamw",
        "sparseadam",
        "adamax",
        "asgd",
        "sgd",
        "lbfgs",
        "rprop",
        "rmsprop",
    ]
)


def test_instantiate_classes() -> None:
    actual = set([conf.split(".yaml")[0] for conf in cs.list("torch/optim")])
    assert not actual ^ expected
