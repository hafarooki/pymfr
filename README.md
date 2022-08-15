# PyMFR

PyMFR is intended as a library gathering open-source implementations of tools to analyze time-series
data related to magnetic flux ropes (MFRs).

Currently, PyMFR has an implementation of the Grad-Shafranov (GS)
automated detection algorithm. The implementation is designed to be very fast and can process a day's worth
of data at a wide range of trial axes and durations in less than a minute.
For a demo of the detection algorithm,
see https://colab.research.google.com/drive/1RbExzbcDsqmo60izQZH_FpenYrYzk9wD?usp=sharing

In the future, PyMFR should have other important tools such as
non-linear force-free cylinder/torus/etc. model fitting,
GS reconstruction, etc.