from dataclasses import dataclass, field
import logging
import os
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


defaults = [
    {"hydra/launcher": "joblib"}
]

@dataclass
class DDPConf:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    backend: str = "gloo"
    init_method: str = MISSING
    world_size: int = 4
    rank: int = 0


cs = ConfigStore.instance()
cs.store(name="ddp", node=DDPConf)


@hydra.main(config_name="ddp")
def main(cfg: DictConfig):
    dist.init_process_group(
        backend=cfg.backend,
        init_method=cfg.init_method,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )
    group = dist.new_group(list(range(cfg.world_size)))
    value = torch.tensor([cfg.rank])
    log.info(f"Rank {cfg.rank} - Value: {value}")
    dist.reduce(value, dst=0, op=dist.ReduceOp.SUM, group=group)
    if cfg.rank == 0:
        average = value / 4.0
        log.info(f"Rank {cfg.rank} - Average: {average}")


if __name__ == "__main__":
    main()
