from hydra_configs.torch.utils.data import DataLoaderConf

print(DataLoaderConf)
# if torch version < 1.7.0, we expect to see: "<class 'hydra_configs.torch_v160.utils.data.DataLoaderConf'>"
# else we expect tp see: "<class 'hydra_configs.torch.utils.data.DataLoaderConf'>"
