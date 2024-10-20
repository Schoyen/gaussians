# Gaussian basis functions
![](https://github.com/Schoyen/gaussians/actions/workflows/python-package.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Repository used for exploration of Gaussian and Hermite Gaussian basis
functions.


## Dependencies

To install this package you first need (assuming you already have Python and pip installed) the Rust compiler. The easiest way to get Rust is found [here](https://www.rust-lang.org/tools/install).
Next, in order to install [gs-lib](https://github.com/Schoyen/gs-lib) via Cargo (as long as this project is private) you need to configure Cargo to use git via the command-line and set up an ssh-key for your Github user.
Edit the file `~/.cargo/config` such that you get these settings:
```bash
$ cat .cargo/config
[net]
git-fetch-with-cli = true
```

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

To install the library you can now run:
```bash
pip install .
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
