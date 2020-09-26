# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from setuptools import find_packages, setup

setup(
    name="hydra-torch-config",
    version="0.9",
    packages=find_packages(include=["gen"]),
    author="Omry Yadan",                # TODO: additional maintainers
    author_email="omry@fb.com",         # TODO: additional maintainers
    url="http://hydra.cc",              # TODO: repo link
    include_package_data=True,
    install_requires=[
        "hydra-configen",
    ],
)
