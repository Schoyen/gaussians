from setuptools import setup, find_packages

setup(
    name="gaussians",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy==1.4.1",
        "numba",
        "numba-scipy @ git+https://github.com/person142/numba-scipy.git#egg=numba-scipy",
    ],
)
