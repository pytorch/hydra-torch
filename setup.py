# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements/requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name="hydra-torch-config",
    version="0.9",
    packages=find_packages(include=["gen"]),
    author="Omry Yadan",  # TODO: additional maintainers
    author_email="omry@fb.com",  # TODO: additional maintainers
    url="http://hydra.cc",  # TODO: repo link
    include_package_data=True,
    install_requires=install_requires,
)