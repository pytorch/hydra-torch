import torch
from packaging import version
import importlib
import re

CURRENT_TORCH_VERSION = "1.7.0"
LEGACY_VERSION_LIST = ["1.6.0"]


def override_imports_for_legacy(module_name):
    module = None
    if version.parse(torch.__version__) < version.parse(CURRENT_TORCH_VERSION):
        legacy_version = LEGACY_VERSION_LIST[0].replace(".", "")
        # regex which replaces the second '.' in the module path with '_vXXX.'
        # EX: 'hydra_configs.torch.foo.bar' -> 'hydra_configs.torch_v160.foo.bar'
        legacy_module_name = re.sub(
            r"^(.*?(\..*?){1})\.", r"\1_v" + legacy_version + ".", module_name
        )
        module = importlib.import_module(legacy_module_name)
    return module
