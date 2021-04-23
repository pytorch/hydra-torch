# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# flake8: noqa
from .optim import register as optim_register
from .nn.modules import register as modules_register
from .utils.data import register as data_register


def register():
    optim_register()
    modules_register()
    data_register()
