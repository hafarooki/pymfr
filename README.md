# PyMFR

PyMFR is intended as a library gathering open-source implementations of tools to analyze time-series
data related to magnetic flux ropes (MFRs).

## Features
* Grad-Shafranov (GS)-based automated detection algorithm (Hu et al. 2018, see also fluxrope.info).
    PyMFR's implementation is very fast, using GPU computation and improvements to the algorithm for performance.
    It can process a month's worth of data at a wide range of trial axes and durations in mere seconds.
    For a demo of the detection algorithm,
    see the demo folder or try it yourself at https://colab.research.google.com/drive/1RbExzbcDsqmo60izQZH_FpenYrYzk9wD?usp=sharing
* Grad-Shafranov (GS) reconstruction (see e.g. Hu & Sonnerup 2002). For an example, see demo/example_reconstruction.ipynb.
* Notebooks with demos of various analytical models
