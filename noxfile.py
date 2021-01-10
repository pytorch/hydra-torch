# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import nox
import os

DEFAULT_PYTHON_VERSIONS = ["3.6", "3.7", "3.8"]
PYTHON_VERSIONS = os.environ.get(
    "NOX_PYTHON_VERSIONS", ",".join(DEFAULT_PYTHON_VERSIONS)
).split(",")

VERBOSE = os.environ.get("VERBOSE", "0")
SILENT = VERBOSE == "0"

# Linted dirs/files:
lint_targets = "."
# Test dirs (corresponds to each project having its own tests folder):
# Note the './', this installs local packages
test_targets = [
    "./" + p.rstrip("\n") for p in open("hydra-configs-projects.txt", "r").readlines()
]


def setup_dev_env(session):
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "setuptools",
        "pip",
        silent=SILENT,
    )

    session.run("pip", "install", "-r", "requirements/dev.txt", silent=SILENT)


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True)
def lint(session):
    setup_dev_env(session)
    session.run("black", *lint_targets, "--check")
    session.run("flake8", "--config", ".flake8", *lint_targets)


@nox.session(python=PYTHON_VERSIONS, reuse_venv=True)
def tests(session):
    setup_dev_env(session)
    for target in test_targets:
        session.run(
            "pip", "install", "-r", target + "/requirements/dev.txt", silent=SILENT
        )
    session.install(*test_targets)  # install config packages
    session.run("pytest", "-v", *test_targets)
