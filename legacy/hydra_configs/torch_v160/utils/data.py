from dataclasses import dataclass
from omegaconf import MISSING
from typing import Any


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
