import hydra
from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
from typing import Any, List
from hydra.core.config_store import ConfigStore

from config.torch.nn.modules import LinearConf, Conv2dConf, Dropout2dConf

@dataclass
class MNISTNetConf:
    defaults: List[Any] = field(default_factory=lambda:
            [{"layer1": "conv1"},
             {"layer2": "conv2"},
             {"layer3": "dropout1"},
             {"layer4": "linear1"},
             {"layer5": "dropout2"},
             {"layer6": "linear2"},
            ])
    layer1: Conv2dConf = MISSING
    layer2: Conv2dConf = MISSING
    layer3: Dropout2dConf = MISSING
    layer4: LinearConf = MISSING
    layer5: Dropout2dConf = MISSING
    layer6: LinearConf = MISSING

cs = ConfigStore.instance()

cs.store(name="config", node=MNISTNetConf)
cs.store(group="layer1", name="conv1", node=Conv2dConf(in_channels=1, out_channels=32, kernel_size=3))
cs.store(group="layer2", name="conv2", node=Conv2dConf(in_channels=32, out_channels=64, kernel_size=3))
cs.store(group="layer3", name="dropout1", node=Dropout2dConf(p=0.25))
cs.store(group="layer4", name="linear1", node=LinearConf(in_features=9216, out_features=128))
cs.store(group="layer5", name="dropout2", node=Dropout2dConf(p=0.5))
cs.store(group="layer6", name="linear2", node=LinearConf(in_features=128, out_features=10))

@hydra.main(config_name="config")
def run(cfg: LinearConf) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    run()
