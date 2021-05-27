from setuptools import setup, find_packages
from setuptools_rust import RustExtension, Binding


setup(
    name="gaussians",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "tqdm",
    ],
    rust_extensions=[
        RustExtension(
            "gaussians.one_dim_lib", "Cargo.toml", binding=Binding.PyO3
        ),
        RustExtension(
            "gaussians.two_dim_lib", "Cargo.toml", binding=Binding.PyO3
        ),
    ],
    zip_safe=False,
)
