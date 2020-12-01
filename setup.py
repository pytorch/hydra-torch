# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from setuptools import setup

projects = [p.rstrip("\n") for p in open("hydra-configs-projects.txt", "r").readlines()]
project_uris = [
    f"{project} @ git+https://github.com/pytorch/hydra-torch/#subdirectory={project}"
    for project in projects
]

setup(
    name="hydra-torch",
    version="0.9",
    author=["Omry Yadan", "Rosario Scalise"],
    author_email=["omry@fb.com", "rosario@cs.uw.edu"],
    url="http://github.com/pytorch/hydra-torch",
    include_package_data=True,
    install_requires=project_uris,
)
