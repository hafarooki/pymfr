# PyMFR

PyMFR is intended as a library gathering open-source implementations of tools to analyze time-series
data related to magnetic flux ropes (MFRs).

Currently, PyMFR has an implementation of the Grad-Shafranov (GS)
automated detection algorithm. The implementation is designed to be very fast and can process a day's worth
of data at a wide range of trial axes and durations in mere seconds.
For a demo of the detection algorithm,
see https://colab.research.google.com/drive/1RbExzbcDsqmo60izQZH_FpenYrYzk9wD?usp=sharing

In the future, PyMFR should have other important tools such as
non-linear force-free cylinder/torus/etc. model fitting,
GS reconstruction, etc.

*New in 1.1*
In addition to the performance improvement from the GPU,
this implementation includes a trick I came up with
and presented at NSRM 2023. Essentially,
the vertical direction of the cross section can be determined analytically,
since \< B \> and V are necessarily perpendicular to it.
This VASTLY reduces the search time since now we need only
check the possible axes that are on the xz plane rather than search the whole 3D space of directions.
