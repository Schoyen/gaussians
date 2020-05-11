from setuptools import setup, find_packages
from setuptools_rust import RustExtension


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
    rust_extensions=[RustExtension("gaussians.gaussian_lib", "./Cargo.toml",),],
    zip_safe=False,
    setup_requires=["setuptools-rust"],
)
