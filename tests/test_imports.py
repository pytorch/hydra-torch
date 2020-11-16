# flake8: noqa

import torch
from hydra_configs.torch.utils.data import DataLoaderConf

print(f"ENV_TORCH_VERSION: {torch.__version__}")
print(f"Imported DataLoaderConf: {DataLoaderConf}")

del DataLoaderConf

from hydra_configs.torch_v160.utils.data import DataLoaderConf

print(f"Imported DataLoaderConf: {DataLoaderConf}")

del DataLoaderConf

import hydra_configs.torch.utils.data

print(f"Imported DataLoaderConf: {hydra_configs.torch.utils.data.DataLoaderConf}")

del hydra_configs.torch.utils.data

import hydra_configs.torch.utils.data as data

print(f"Imported DataLoaderConf: {data.DataLoaderConf}")

# if torch version < 1.7.0, we expect to see: "<class 'hydra_configs.torch_v160.utils.data.DataLoaderConf'>"
# else we expect tp see: "<class 'hydra_configs.torch.utils.data.DataLoaderConf'>"
