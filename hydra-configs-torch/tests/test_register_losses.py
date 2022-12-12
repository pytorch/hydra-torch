# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_store import ConfigStore
import hydra_configs.torch.nn.modules

cs = ConfigStore()


# registers all 'base' optimizer configs with configstore instance
hydra_configs.torch.nn.modules.register()

expected = set(
    [
        "bceloss",
        "bcewithlogitsloss",
        "cosineembeddingloss",
        "ctcloss",
        "l1loss",
        "hingeembeddingloss",
        "kldivloss",
        "marginrankingloss",
        "mseloss",
        "multilabelmarginloss",
        "multilabelsoftmarginloss",
        "multimarginloss",
        "nllloss",
        "nllloss2d",
        "poissonnllloss",
        "smoothl1loss",
        "softmarginloss",
        "tripletmarginloss",
    ]
)


def test_instantiate_classes() -> None:
    actual = set([conf.split(".yaml")[0] for conf in cs.list("torch/nn/modules/loss")])
    assert not actual ^ expected
