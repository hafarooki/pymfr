from setuptools import setup

try:
    import torch
except ImportError:
    raise Exception("pymfr requires PyTorch")

setup(
    name="pymfr",
    version="1.4.1",
    author="Hameedullah A. Farooki",
    author_email="haf5@njit.edu",
    packages=["pymfr"],
    keywords="pymfr fluxrope mfr smfr grad-shafranov gs",
    install_requires=["torch>=1.8.2",
                      "numpy",
                      "scipy",
                      "tqdm",
                      "torchinterp1d @ git+https://github.com/aliutkus/torchinterp1d.git"]
)
