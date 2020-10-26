# MNIST Basic Tutorial

This tutorial series is built around the [PyTorch MNIST example] and is meant to demonstrate how to modify your PyTorch code to be configured by Hydra. We will start with the simplest case which introduces one central concept while minimizing altered code. In the following tutorials ([Intermediate][Intermediate Tutorial] and [Advanced][Advanced Tutorial]), we will show how a few additional changes can yield an even more powerful end product.

The source file can be found at [mnist_00.py].

### Pre-reading
Although this tutorial is aimed at being self-contained, taking a look through Hydra's terminology as well as the basic and advanced tutorials couldn't hurt.

1. [Hydra Terminology]
2. [Hydra Basic Tutorial]
3. [Hydra Structured Configs Tutorial]

### Contents

1. [The Hydra Block](#the-hydra-block)
    1. [Imports](#imports)
    2. [Parting with Argparse](#parting-with-argparse)
    3. [Top Level Config](#top-level-config)
    4. [Adding the Top Level Config to the ConfigStore](#adding-the-top-level-config-to-the-configstore)
2. [Dropping into `main()`](#dropping-into-main)
	1. [Instantiating the Optimizer and Scheduler](#instantiating-the-optimizer-and-scheduler)
3. [Running with Hydra](#running-with-hydra)
	1. [Commandline Overrides](#command-line-overrides)
	2. [Multirun](#multirun)
4. [Summary](#summary)


***
## The 'HYDRA BLOCK'

For clarity, as we modify the [PyTorch MNIST example], we will make the diffs explicit. Most of the changes we introduce will be at the top of the file within the commented `##### HYDRA BLOCK #####`, though in practice much of this block could reside in its own concise imported file.

### Imports
```python
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

# hydra-torch structured config imports
from config.torch.optim import AdadeltaConf
from config.torch.optim.lr_scheduler import StepLRConf
```

There are two areas in our Hydra-specific imports. First, since we define configs in this file, we need access to the following:
- the `ConfigStore`
- the `dataclass` decorator (for structured configs)

**The [ConfigStore]** is a singleton object which all config objects are registered to. This gives Hydra access to our structured config definitions *once they're registered*.

**[Structured Configs][hydra structured configs tutorial]** are dataclasses that Hydra can use to compose complex config objects. We can think of them as templates or 'starting points' for our configs. Each `*Conf` file provided by `hydra-torch` is a structured config. See an example of one below:

```python
# the structured config for Adadelta imported from config.torch.optim:
@dataclass
class AdadeltaConf:
    _target_: str = "torch.optim.adadelta.Adadelta"
    params: Any = MISSING
    lr: Any = 1.0
    rho: Any = 0.9
    eps: Any = 1e-06
    weight_decay: Any = 0
```

> **NOTE:** [`MISSING`] is a special constant used to indicate there is no default value specified.

The second set of imports correspond to two components in the training pipeline of the [PyTorch MNIST example]:

- `Adadelta` which resides in `torch.optim`
- `StepLR` which resides in `torch.optim.lr_scheduler`

Note that the naming convention for the import hierarchy mimics that of `torch`. We correspondingly import the following structured configs:
- `AdadeltaConf` from `config.torch.optim`
- `StepLRConf` from `config.torch.optim.lr_scheduler`

Generally, we follow the naming convention of applying the suffix `-Conf` to distinguish the structured config class from the class of the object to be configured.

***
### Top Level Config
After importing two pre-defined structured configs for components in our training pipeline, the optimizer and scheduler, we still need a "top level" config to merge everything. We can call this config class `MNISTConf`. You will notice that this class is nothing more than a python `dataclass` and corresponds to, you guessed it, a *structured config*.

> **NOTE:** The top level config is application specific and thus is not provided  by `hydra-torch`.

We can start this out by including the configs we know we will need for the optimizer (`Adadelta`) and scheduler (`StepLR`):
```python
# our top level config:
@dataclass
class MNISTConf:
    adadelta: AdadeltaConf = AdadeltaConf()
    steplr: StepLRConf = StepLRConf(step_size=1)
```
Notice that for `StepLRConf()` we need to pass `step_size=1` when we initialize because it's default value is `MISSING`.
```python
# the structured config imported from hydra-torch in config.torch.optim.lr_scheduler
@dataclass
class StepLRConf:
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    optimizer: Any = MISSING
    step_size: Any = MISSING
    gamma: Any = 0.1 last_epoch: Any = -1
```
> **NOTE:** The `hydra-torch` configs are generated from the PyTorch source and rely on whether the module uses type annotation. Once additional type annotation is added, these configs will become more strict providing greater type safety.

Later, we will specify the optimizer (also default `MISSING`) as a passed through argument when the actual `StepLR` object is instantiated.

### Adding the Top Level Config to the ConfigStore
Very simply, but crucially, we add the top-level config class `MNISTConf` to the `ConfigStore` in two lines:
```python
cs = ConfigStore.instance()
cs.store(name="mnistconf", node=MNISTConf)
```
The name `mnistconf` will be passed to the `@hydra` decorator when we get to `main()`.

***
### Parting with Argparse

Now we're starting to realize our relationship with `argparse` isn't as serious as we thought it was. Although `argparse` is powerful, we can take it a step further. In the process we hope to introduce greater organization and free our primary file from as much boilerplate as possible.

One feature Hydra provides us is aggregating our configuration files alongside any 'specifications' we pass via command line arguments. What this means is as long as we have the configuration file which defines possible arguments like `save_model` or `dry_run`, there is no need to also litter our code with `argparse` definitions.

This whole block in `main()`:
```python
def main():
# Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
```
becomes:
```python
def main(cfg):
# All argparse args now reside in cfg
```
Our initial strategy is to dump these arguments directly in our top-level configuration.
```python
@dataclass
class MNISTConf:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 14
    no_cuda: bool = False
    dry_run: bool = False
    seed: int = 1
    log_interval: int
    save_model: bool = False
    adadelta: AdadeltaConf = AdadeltaConf()
    steplr: StepLRConf = StepLRConf(step_size=1)
```
> **NOTE:** `learning_rate` and `gamma` are included in `AdadeltaConf()` and so they were omitted from the top-level args.

This works, but can feel a bit flat and disorganized (much like `argparse` args can be). Don't worry, we will remedy this later in the tutorial series. Note, we also sacrifice `help` strings. This is a planned feature, but not supported in Hydra just yet.

Now our `argparse` args are at the same level as our optimizer and scheduler configs. We will remove `lr` and `gamma` since they are already present within the optimizer config `AdadeltaConf`.
***
## Dropping into `main()`
Now that we've defined all of our configs, we just need to let Hydra create our `cfg` object at runtime and make sure the `cfg` is plumbed to any object we want it to configure.
```python
@hydra.main(config_name='mnistconf')
def main(cfg):
    print(cfg.pretty())
    ...
```
The single idea here is that `@hydra.main` looks for a config in the `ConfigStore` instance, `cs` named "`mnistconf`". It finds the `MNISTConf` (our top level conf) we registered to that name and populates `cfg` inside `main()` with the fully expanded structured config. This includes our optimizer and scheduler configs, `cfg.adadelta` and `cfg.steplr`, respectively.

Instrumenting `main()` is simple. Anywhere we find `args`, replace this with `cfg` since we put all of the `argparse` arguments at the top level. For example, `args.batch_size` becomes `cfg.batch_size`:
```python
# the first few lines of main
    ...
    use_cuda = not cfg.no_cuda and torch.cuda.is_available() # DIFF args.no_cuda
        torch.manual_seed(cfg.seed) # DIFF args.seed
        device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {'batch_size': cfg.batch_size} # DIFF args.batch_size
        test_kwargs = {'batch_size': cfg.test_batch_size} # DIFF args.test_batch_size
    ...
```


### Instantiating the optimizer and scheduler
Still inside `main()`, we want to draw attention to two slightly special cases before moving on. Both the `optimizer` and `scheduler` are instantiated manually by specifying each argument with its `cfg` equivalent. Note that since these are nested fields, each of these parameters is two levels down e.g. `lr=args.learning_rate` becomes `lr=cfg.adadelta.lr`.

```python
 optimizer = Adadelta(lr=cfg.adadelta.lr, #DIFF lr=args.learning_rate
                         rho=cfg.adadelta.rho,
                         eps=cfg.adadelta.eps,
                         weight_decay=cfg.adadelta.weight_decay,
                         params=model.parameters()
 ```
In this case, the `optimizer` has one argument that is not a part of our config -- `params`. If it wasn't obvious, this needs to be passed from the initialized `Net()` called model. In the structured config that initialized `cfg.adadelta`, `params` is default to `MISSING`. The same is true of the `optimizer` field in `StepLRConf`.

```python
scheduler = StepLR(step_size=cfg.steplr.step_size,
                       gamma=cfg.steplr.gamma,
                       last_epoch=cfg.steplr.last_epoch,
                       optimizer=optimizer
 ```
 This method for instantiation is the least invasive to the original code, but it is also the least flexible and highly verbose. Check out the [Intermediate Tutorial] for a better approach that will allow us to hotswap optimizers and schedulers, all while writing less code.

***
## Running with Hydra

```bash
$ python 00_minst.py
```
That's it. Since the `@hydra.main` decorator is above `def main(cfg)`, Hydra will manage the command line, logging, and saving outputs to a date/time stamped directory automatically. These are all configurable, but the default behavior ensures expected functionality. For example, if a model checkpoint is saved, it will appear in a new directory `./outputs/DATE/TIME/`.

### New Super Powers 🦸

#### Command Line Overrides

Much like passing argparse args through the CLI, we can use our default values specified in `MNISTConf` and override only the arguments/parameters we want to tweak:

```bash
$  python mnist_00.py epochs=1 save_model=True checkpoint_name='experiment0.pt'
```

For more on command line overrides, see: [Hydra CLI] and [Hydra override syntax].

#### Multirun
We often end up wanting to sweep our optimizer's learning rate. Here's how Hydra can help facilitate:
```bash
$ python mnist_00.py -m adadelta.lr="0.001, 0.01, 0.1"
```
Notice the `-m` which indicates we want to schedule 3 jobs where the learning rate changes by an order of magnitude across each training session.

It can be useful to test multirun outputs by passing `dry_run=True` and setting `epochs=1`:
```bash
$ python mnist_00.py -m epochs=1 dry_run=True adadelta.lr="0.001,0.01, 0.1"
```

> **NOTE:** these jobs can be dispatched to different resources and run in parallel or scheduled to run serially (by default). More info on multirun: [Hydra Multirun]. Hydra can use different hyperparameter search tools as well. See: [Hydra Ax plugin] and [Hydra Nevergrad plugin].

***
## Summary
In this tutorial, we demonstrated the path of least resistance to configuring your existing PyTorch code with Hydra. The main benefits we get from the 'Basic' level are:
- No more boilerplate `argparse` taking up precious linecount.
- All training related arguments (`epochs`, `save_model`, etc.)  are now configurable via Hydra.
- **All** optimizer/scheduler (`Adadelta`/`StepLR`) arguments are exposed for configuration
  -- extending beyond only the ones the user wrote argparse code for.
- We have offloaded the book-keeping of compatible `argparse` code to Hydra via `hydra-torch` which runs tests ensuring all arguments track the API for the correct version of `pytorch`.

However, there are some limitations in our current strategy that the [Intermediate Tutorial] will address. Namely:
- Configuring the model (*think architecture search*)
- Configuring the dataset (*think transfer learning*)
- Swapping in and out different Optimizers/Schedulers

Once comfortable with the basics, continue on to the [Intermediate Tutorial].

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [pytorch mnist example]: <https://github.com/pytorch/examples/blob/master/mnist/main.py>
   [mnist_00.py]: mnist_00.py
   [config schema]: <https://hydra.cc/docs/tutorials/structured_config/schema>
   [configstore]: <https://hydra.cc/docs/tutorials/structured_config/config_store>
   [hydra basic tutorial]: <https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli>
   [hydra structured configs tutorial]: <https://hydra.cc/docs/tutorials/structured_config/intro>
   [hydra structured configs example]: <https://hydra.cc/docs/tutorials/structured_config/minimal_example>
   [hydra terminology]: <https://hydra.cc/docs/terminology>
   [omegaconf]: <https://omegaconf.readthedocs.io/en/latest/>
   [`missing`]: <https://omegaconf.readthedocs.io/en/latest/structured_config.html?highlight=MISSING#mandatory-missing-values>
   [hydra cli]: <https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli>
   [hydra override syntax]: <https://hydra.cc/docs/advanced/override_grammar/basic>
   [hydra multirun]: <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run>
   [hydra ax plugin]: <https://hydra.cc/docs/plugins/nevergrad_sweeper>
   [hydra nevergrad plugin]: <https://hydra.cc/docs/plugins/nevergrad_sweeper>
   [Intermediate Tutorial]: mnist_01.md
   [Advanced Tutorial]: mnist_02.md
