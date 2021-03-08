# hydra-torch
Configuration classes enabling type-safe PyTorch configuration for Hydra apps.  
**This repo is work in progress.**

The config dataclasses are generated using [configen](https://github.com/facebookresearch/hydra/tree/master/tools/configen), check it out if you want to generate config dataclasses for your own project.

### Install:
```
# For now, please obtain through github. Soon, versioned (per-project) dists will be on PyPI.
pip install git+https://github.com/pytorch/hydra-torch
```

### Example config:
Here is one of many configs available. Notice it uses the defaults defined in the torch function signatures:
```python
@dataclass
class TripletMarginLossConf:
    _target_: str = "torch.nn.modules.loss.TripletMarginLoss"
    margin: float = 1.0
    p: float = 2.0
    eps: float = 1e-06
    swap: bool = False
    size_average: Any = None
    reduce: Any = None
    reduction: str = "mean"
```

### Importing Convention:
```python
from hydra_configs.<package_name>.path.to.module import <ClassName>Conf
```
where `<package_name>` is the package being configured and `path.to.module` is the path in the original package.

Inferring where the package is located is as simple as prepending `hydra_configs.` and postpending `Conf` to the original class import:
e.g.
```python
#module to be configured
from torch.optim.adam import Adam

#config for the module
from hydra_configs.torch.optim.adam import AdamConf
```


### Getting Started:
Take a look at our tutorial series:
1. [Basic Tutorial](examples/mnist_00.md)
2. Intermediate Tutorial (coming soon)
3. Advanced Tutorial (coming soon)

### Other Config Projects:
A list of projects following the `hydra_configs` convention (please notify us if you have one!):

[Pytorch Lightning](https://github.com/romesco/hydra-lightning)

### License
hydra-torch is licensed under [MIT License](LICENSE).
