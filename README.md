# hydra-torch
Configuration classes enabling type-safe PyTorch configuration for Hydra apps.  
**This repo is work in progress.**

The config dataclasses are generated using [configen](https://github.com/facebookresearch/hydra/tree/master/tools/configen), check it out if you want to generate config dataclasses for your own project.

### Example config:
Here is one of many configs available. Notice it uses the defaults defined in the torch function signatures:
```python
@dataclass
class AdadeltaConf:
    _target_: str = "torch.optim.adadelta.Adadelta"
    params: Any = MISSING
    lr: Any = 1.0
    rho: Any = 0.9
    eps: Any = 1e-06
    weight_decay: Any = 0
```

### Getting Started:
Take a look at our tutorial series:
1. [Basic Tutorial](examples/mnist_00.md)
2. Intermediate Tutorial (coming soon)
3. Advanced Tutorial (coming soon)

### License
hydra-torch is licensed under [MIT License](LICENSE).
