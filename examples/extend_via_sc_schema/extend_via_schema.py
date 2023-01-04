# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore
from hydra_configs.torch.optim import AdamConf


@dataclass
class Config:
    optim: AdamConf
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="optim", name="base_adam", node=AdamConf)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
