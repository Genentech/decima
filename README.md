[![PyPI-Server](https://img.shields.io/pypi/v/decima.svg)](https://pypi.org/project/decima/)
![Unit tests](https://github.com/genentech/decima/actions/workflows/run-tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/870361048.svg)](https://doi.org/10.5281/zenodo.15319897)

# Decima

## Introduction
Decima is a Python library to train sequence models on single-cell RNA-seq data.

![Figure](assets/fig1.png)

## Weights
Weights of the trained Decima models (4 replicates) are now available at https://zenodo.org/records/15092691. See the tutorial for how to load and use these.

## Preprint
Please cite https://www.biorxiv.org/content/10.1101/2024.10.09.617507v3. Also see https://github.com/Genentech/decima-applications for all the code used to train and apply models in this preprint.

## Requirements
Decima has been tested on Ubuntu 24.04.3 and MacOS 15.6.1 using Python 3.9-3.12.

## Installation

Install the package from PyPI,

```sh
pip install decima
```

Or if you want to be on the cutting edge,

```sh
pip install git+https://github.com/genentech/decima.git@main
```
Typical installation time including all dependencies is under 10 minutes.

## Tutorials
See the [tutorials](docs/tutorials) for instructions, including how to train your own Decima model with an example dataset.

<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).
