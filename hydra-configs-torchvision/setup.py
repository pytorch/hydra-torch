# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from setuptools import find_namespace_packages, setup

requirements = [
    "omegaconf",
]

setup(
    name="hydra-configs-torchvision",
    version="0.8.2",
    packages=find_namespace_packages(include=["hydra_configs*"]),
    author=["Omry Yadan", "Rosario Scalise"],
    author_email=["omry@fb.com", "rosario@cs.uw.edu"],
    url="http://github.com/pytorch/hydra-torch",
    include_package_data=True,
    install_requires=requirements,
)
