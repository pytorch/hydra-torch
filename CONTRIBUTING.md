# Contributing to hydra-torch
We want to make contributing to this project as easy and transparent as
possible.

To see a list of config projects currently maintained in this repository, please see: [hydra-configs-projects.txt](hydra-configs-projects.txt)

In order to track progress or find an issue to draft a PR for, please see the [**Projects**](https://github.com/pytorch/hydra-torch/projects) tab.

## Opportunities to Contribute
There are 3 main ways to contribute starting with the most straightfoward option:

1. **Filing Issues against Configs:** Noticing a bug that you believe is a problem with the generated config? Please file an issue for the offending config class stating clearly your versions of `hydra-core` and the library being configured (e.g. `torch`). If you believe it is actually a problem with hydra or torch, file the issue in their respective repositories. 

> **NOTE:** Please include the manual tag for the project and library version in your issue title. If, for example, there is an issue with `AdamConf` for `torch1.6` which comes from `hydra-configs-torch` v1.6.1, your issue name might look like:
**[hydra-configs-torch][1.6.1] AdamConf does not instantiate**.

2. **Example Usecase / Tutorial:** The `hydra-torch` repository not only hosts config packages like `hydra-configs-torch`,`hydra-configs-torchvision`, etc., it also aggregates examples of how to structure projects utilizing hydra and torch. The bar is high for examples that get included, but we will work together as a community to hone in on what the best practices are. Ideally, example usecases will come along with an incremental tutorial that introduces a user to the methodology being followed. If you have an interesting way to use hydra/torch, write up an example and show us in a draft PR!

3. **Maintaining Configs:** After the initial (considerable) setup effort, the goal of this repository is to be self-sustaining meaning code can be autogrenerated when APIs change. In order to contribute to a particular package like `hydra-configs-torch`, please see the [**Projects**](https://github.com/pytorch/hydra-torch/projects) tab to identify outstanding issues per project and configured library version. Before contributing at this level, please familiarize with [configen](https://github.com/facebookresearch/hydra/tree/master/tools/configen). We are actively developing this tool as well.

## Linting / Formatting
Please download the formatting / linting requirements: `pip install -r requirements/dev.txt`.
Please install the pre-commit config for this environment: `pre-commit install`.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to hydra-torch, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
