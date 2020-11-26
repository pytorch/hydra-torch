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

# from .optimizer import OptimizerConf
from .lbfgs import LBFGSConf
from . import lr_scheduler

# from . import swa_utils

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
# del optimizer
del lbfgs
