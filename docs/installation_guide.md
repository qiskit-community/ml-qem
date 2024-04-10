# Blackwater Installation Guide

## PyPi

PyPi installation is not available yet.

## Local installation

1. Clone repo
2. Create and activate new python env
```shell
conda create -n mlqem python=3.9
```

3. Install pytorch https://pytorch.org/get-started/locally/#start-locally
4. Install other dependencies

```shell
pip install -r requirements.txt
```

5. Installing Blackwater

```shell
pip install -e .
```

Estimated time to install all dependencies: <5 minutes