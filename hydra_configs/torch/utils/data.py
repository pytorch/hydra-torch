from dataclasses import dataclass
from omegaconf import MISSING
from typing import Any
import torch
from packaging import version
import importlib


def override_imports_for_legacy():
    if version.parse(torch.__version__) < version.parse("1.6.0"):
        module = importlib.import_module("hydra_configs.torch_v160.utils.data")
        globals().update(
            {n: getattr(module, n) for n in module.__all__}
            if hasattr(module, "__all__")
            else {k: v for (k, v) in module.__dict__.items() if not k.startswith("_")}
        )


@dataclass
class DataLoaderConf:
    """For more details on parameteres please refer to the original documentation:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    _target_: str = "torch.utils.data.DataLoader"
    dataset: Any = MISSING
    batch_size: Any = 1
    shuffle: Any = False
    sampler: Any = None
    batch_sampler: Any = None
    num_workers: Any = 0
    collate_fn: Any = None
    pin_memory: Any = False
    drop_last: Any = False
    timeout: Any = 0
    worker_init_fn: Any = None
    multiprocessing_context: Any = None
    generator: Any = None


override_imports_for_legacy()
