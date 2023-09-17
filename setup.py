from setuptools import setup

try:
    import torch
except ImportError:
    raise Exception("pymfr requires PyTorch")

setup(
    name="pymfr",
    version="2.0.0",
    author="Hameedullah A. Farooki",
    author_email="haf5@njit.edu",
    packages=["pymfr"],
    keywords="pymfr fluxrope mfr smfr grad-shafranov gs",
    install_requires=["torch",
                      "numpy",
                      "scipy",
                      "tqdm",
                      "xarray",
                      "torchinterp1d @ git+https://github.com/aliutkus/torchinterp1d.git"]
)
