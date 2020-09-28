# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import nox
import os

DEFAULT_PYTHON_VERSIONS = ["3.6", "3.7", "3.8"]
PYTHON_VERSIONS = os.environ.get(
    "NOX_PYTHON_VERSIONS", ",".join(DEFAULT_PYTHON_VERSIONS)
).split(",")

VERBOSE = os.environ.get("VERBOSE", "0")
SILENT = VERBOSE == "0"

# Linted & Formatted dirs/files:
targets = "config", "tests", "noxfile.py"


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


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    setup_dev_env(session)
    session.run("flake8", "--config", ".flake8", *targets)


@nox.session(python=PYTHON_VERSIONS)
def black(session):
    setup_dev_env(session)
    session.run("black", *targets)
