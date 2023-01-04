##Leftovers
There are two areas in our Hydra-specific imports. First, since we define configs in this file, we need access to the following:
- typing from both `typing` and `omegaconf`

**[OmegaConf]** is an external library that Hydra is built around. Every config object is a datastructure defined by OmegaConf. For our purposes, we use it to specify typing and special constants such as [`MISSING`] when there is no value specified.

#### Config Schema
*our config templates - providing type checking and good defaults*

Second, we import two [config schema] from `hydra-torch`. Think of config schema as recommended templates for commonly used configurations. `hydra-torch` provides config schema for a large subset of common PyTorch classes. In the basic tutorial, we only consider the schema for the PyTorch classes:
- `Adadelta` which resides in `torch.optim`
- `StepLR` which resides in `torch.optim.lr_scheduler`
