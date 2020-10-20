# This file mirrors the __init__.py within pytorch/torch/optim
# It allows hydra-torch configs to be imported in the same style
# flake8: noqa

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

# from . import lr_scheduler

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
