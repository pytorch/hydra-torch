# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# flake8: noqa
# Mirrors torch/optim __init__ to allow for symmetric import structure
from .adadelta import AdadeltaConf
from .adagrad import AdagradConf
from .adam import AdamConf
from .adamw import AdamWConf
from .sparse_adam import SparseAdamConf
from .adamax import AdamaxConf
from .asgd import ASGDConf
from .sgd import SGDConf
from .rprop import RpropConf
from .rmsprop import RMSpropConf

from .lbfgs import LBFGSConf
from . import lr_scheduler

from hydra.core.config_store import ConfigStore


def register_configs():
    cs = ConfigStore.instance()
    cs.store(provider="torch", group="torch/optim", name="adadelta", node=AdadeltaConf)
    cs.store(provider="torch", group="torch/optim", name="adagrad", node=AdagradConf)
    cs.store(provider="torch", group="torch/optim", name="adam", node=AdamConf)
    cs.store(provider="torch", group="torch/optim", name="adamw", node=AdamWConf)
    cs.store(
        provider="torch", group="torch/optim", name="sparseadam", node=SparseAdamConf
    )
    cs.store(provider="torch", group="torch/optim", name="adamax", node=AdamaxConf)
    cs.store(provider="torch", group="torch/optim", name="asgd", node=ASGDConf)
    cs.store(provider="torch", group="torch/optim", name="sgd", node=SGDConf)
    cs.store(provider="torch", group="torch/optim", name="lbfgs", node=LBFGSConf)
    cs.store(provider="torch", group="torch/optim", name="rprop", node=RpropConf)
    cs.store(provider="torch", group="torch/optim", name="rmsprop", node=RMSpropConf)


del adadelta
del adagrad
del adam
del adamw
del sparse_adam
del adamax
del asgd
del sgd
del rprop
del rmsprop
del lbfgs
