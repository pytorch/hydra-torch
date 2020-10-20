# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pathlib

import pkg_resources
from setuptools import find_namespace_packages, setup

with pathlib.Path("requirements/requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name="hydra-torch-config",
    version="0.9",
    packages=find_namespace_packages(include=["config.*"]),
    author=["Omry Yadan", "Rosario Scalise"],
    author_email=["omry@fb.com", "rosario@cs.uw.edu"],
    url="http://github.com/pytorch/hydra-torch",
    include_package_data=True,
    install_requires=install_requires,
)
