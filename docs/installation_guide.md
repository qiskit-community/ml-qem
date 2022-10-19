# Blackwater Installation Guide

## PyPi

```shell
pip install blackwater
```

## Local installation

1. Clone repo
2. Installing Depencencies

```shell
pip install -r requirements.txt
```

3. Installing Optional Dependencies

```shell
pip install -r requirements-dev.txt
```
4. Installing Blackwater

```shell
pip install .
```

5. Testing the Installation

```shell
tox -epy39
tox -elint
```