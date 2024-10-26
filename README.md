# Gaussian basis functions
![](https://github.com/Schoyen/gaussians/actions/workflows/python-package.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Repository used for exploration of Gaussian and Hermite Gaussian basis
functions.


## Dependencies

To install this package you first need (assuming you already have Python and pip installed) the Rust compiler. The easiest way to get Rust is found [here](https://www.rust-lang.org/tools/install).

To install the necessary development dependencies using pip , run:
```bash
$ pip install -r requirements.txt
```

### Virtual environment
It is recommended to set up a virtual environment prior to running pip.
Do this via:
```bash
$ python -m venv venv
$ source venv/bin/activate
```
To deactivate the environment run:
```bash
$ deactivate
```

## Installation

If you have cloned the repository you can now run:
```bash
pip install .
```
or
```bash
pip install -e .
```
in your local clone.
Otherwise, you can also install via git by (if an SSH key has been set up):
```bash
pip install git+ssh://git@github.com/Schoyen/gaussians.git
```
Or using regular HTTPS:
```bash
pip install git+https://github.com/Schoyen/gaussians.git
```
