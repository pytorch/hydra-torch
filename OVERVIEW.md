# Overview

The audience for this library is Hydra users using PyTorch and related libraries (torchvision, ...)
The goals of this repository are:

1. Provide a maintained and tested implementation of config classes that can be used instantiate and configure various classes from the supported projects.
2. Provide examples and tutorials demonstrating best practices for using Hydra to configure PyTorch deep learning applications.
3. Showcase a recommended approach for creating other similar configuration packages for other libraries.

1. It demonstrates a standardization for creating and organizing Hydra configuration classes for the ML/torch community and beyond.
2. It unifies a collection of classes across projects in one place, ensuring robust testing, version compatibility, and PyPI distributed packaging for each.
    - We see many hydra users reimplementing these classes (and not tracking APIs of the configured projects). `hydra-torch` factors this code out.
3. It provides best practices and guidance on how to organize code and utilize Hydra for research, production, and other innovative use cases.

##### Terminology for this overview:
- The overall effort and repo is named:`hydra-torch`
- **Library:** The library being configured.
  - e.g. `torch`, `torchvision`
- **Project:** The project corresponding to the library being configured.
  - e.g. `hydra-configs-torch`, `hydra-configs-torchvision`
- **Package:** The installable package containing the configs. Usually corresponding to a specific project.
  - here, also `hydra-configs-torch`, `hydra-configs-torchvision`. Find the package definition in the project dir's `setup.py`.



### Repo Structure
```
📂 hydra-torch
└ 📁 configen 				# source files used when generating package content. all projects have their own subdirectory here.
└ 📁 hydra-configs-torch		# a project corresponding to configuring a specific library (contains package definition)
└ 📁 hydra-configs-torchvision		# "
└ 📁 hydra-configs-<future-library> 	# "
└ 📁 examples				# use-cases and tutorials
```

Each `hydra-configs-<library-name>` defines its own package corresponding to a project it provides classes for. That is, `hydra-configs-torch` contains a package (with its own `setup.py`) which provides classes for `torch`.


### Packaging

This repo maintains multiple packages. An important area of consideration for these packages is to organize them such that they are compatible with each other and have an 'expected', intuitive structure. The easiest way to do this is to enforce a standard for all Hydra configs packages to follow.

Our approach makes use of [Native Namespace Packages](https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages). This allows all packages to share the top level namespace `hydra_configs`. The only real requirement for this package format is to ensure there is **no**  `__init__.py` located in the namespace folder.

The format for folder structure per project is always:
```
📂 hydra-configs-<library-name>     # the dir containing the package of configs for <library>
├ 📁 hydra_configs		    # the namespace we will always use
│ └ 📁 <library-name>		    # e.g. 'torchvision'
│   └ 📁 <library-subpackage-name>  # e.g. 'transforms'
│      ⋮
│      └ <module>.py                # e.g. 'transforms.py'
└ setup.py 			    # configures this package for setuptools
```

The beauty of this approach is that users can be sure the importing idiom is reliably:
```python
from hydra_configs.<project_name>.path.to.module import <ClassName>Conf
```
For example:
```python
from hydra_configs.torch.optim import AdamConf
```
 

This will retain compatibility even with repositories that define configs for hydra outside of the `hydra-torch` repo as long as they follow this pattern.

##### Metapackage

For convenience, we also provide a metapackage called `hydra-torch`. This is defined at the top level of the repo in `setup.py`. Effectively, all this means is that a user can either install each project-specific package as needed or they can obtain them all in one shot by installing only the metapackage.

### Testing

All configs must be tested to ensure they are compatible with the latest version of Hydra. At a minimum, each config must be able to successfully yield the desired class when passed to `hydra.utils.instantiate()`. If the class has special functionality, consider making evaluating this a goal of the test suite as well.

Tests are run through pytest. The CI flow is currently: `CircleCI` -> `nox` -> `pytest`. All tests must pass before anything can be merged.


### Versions

##### Regarding Libraries

Given the rapid development of libraries in the torch ecosystem, it is likely that there will be a need for multiple simultaneous version specific `hydra-configs-<library-name>` per library. This is especially relevant as API and type annotation is updated moving forward for a particular library.

For each library, there will be releases for each `MINOR` version starting with the latest version available at the time of creation of the configs project.
 1. There will be a branch/tag for each version for each library's corresponding configs project. These branches/tags will only contain the content relevant to that particular project.
 2. In `master`, each configs project will maintain the latest supported library version.

###### Package/Branch Names:
The version for the `hydra-configs-<library-name>` package will be `MAJOR.MINOR.X` where `MAJOR.MINOR` tracks the library versions and `.X` is reserved for revisions of the configs for that particular `MAJOR.MINOR` should they be required.

e.g. `hydra-configs-torch==1.6.0` corresponds to `torch==1.6` and if updates are needed for patches from either end, only the `PATCH` version will be updated -> `hydra-config-torch==1.6.1`

#TODO: write out exact tag format once decided upon

##### Regarding Hydra

Many of the features that make using configs compelling are supported in Hydra > 1.1. For this reason, we are aiming support starting at 1.1 . Using an older version of Hydra is not advised, but may still work fine in less complex cases.


### Releases
Releases will be on a per-project basis for specific versions of a project. Releases should be tied to a specific branch/tag in the repo that passes all tests via CI. We release via PyPI.
