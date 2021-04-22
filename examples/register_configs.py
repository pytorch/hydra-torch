# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from typing import Any
from omegaconf import OmegaConf

# registers all 'base' optimizer configs with configstore instance
import hydra_configs.torch.optim

hydra_configs.torch.optim.register_configs()


@hydra.main(config_name="torch/optim/adam")
def my_app(cfg: Any) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
