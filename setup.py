# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pathlib
import pkg_resources
from setuptools import find_namespace_packages, setup
import subprocess

INSTALL_LEGACY_CONFIGS = True

with pathlib.Path("requirements/requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name="hydra-torch-configs",
    version="0.9",
    packages=find_namespace_packages(where=".", include=["hydra_configs*"]),
    namespace_packages=["hydra_configs"],
    author=["Omry Yadan", "Rosario Scalise"],
    author_email=["omry@fb.com", "rosario@cs.uw.edu"],
    url="http://github.com/pytorch/hydra-torch",
    include_package_data=True,
    install_requires=install_requires,
)

# install legacy configs
if INSTALL_LEGACY_CONFIGS:
    subprocess.call(["python", "./legacy/setup.py", "install"])
