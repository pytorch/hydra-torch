import torch
from packaging import version
import importlib
import re
import warnings

CURRENT_SUPPORTED_TORCH_VERSION = "1.7.0"
LEGACY_VERSION_LIST = ["1.6.0"]


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def override_imports_for_legacy(module_name):
    env_torch_version = torch.__version__
    module = None
    if version.parse(env_torch_version) != version.parse(
        CURRENT_SUPPORTED_TORCH_VERSION
    ):
        if env_torch_version in LEGACY_VERSION_LIST:
            legacy_version = env_torch_version.replace(".", "")
            # regex which replaces the second '.' in the module path with '_vXXX.'
            # EX: 'hydra_configs.torch.foo.bar' -> 'hydra_configs.torch_v160.foo.bar'
            legacy_module_name = re.sub(
                r"^(.*?(\..*?){1})\.", r"\1_v" + legacy_version + ".", module_name
            )
            module = importlib.import_module(legacy_module_name)
        else:
            warnings.warn(
                f"Your version of torch is {env_torch_version}, but currently only versions {[CURRENT_SUPPORTED_TORCH_VERSION] + LEGACY_VERSION_LIST} are supported. The {CURRENT_SUPPORTED_TORCH_VERSION} configs will be loaded, but please note the potential for some incompatibility."  # noqa: E501
            )
    return module
