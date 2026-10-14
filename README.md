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

## Version History
- Version 1: GS detection with performance improvements 
    - 1.0.0: Initial implementation of the Hu+18 algorithm (Aug 2022)
    - 1.1.0: Trick to determine the vertical (y) direction**: found analytically as the direction perpendicular to both the mean field and the HT frame velocity, so the axis search is a 1D sweep in the plane perpendicular to y** (Feb 2023)
    - 1.1.1: Removed a no-longer-needed requirement on residue minimization for the potential (Feb 2023)
    - 1.1.2: Pressure is no longer used to determine the axis orientation (Feb 2023)
    - 1.1.3: Code clarity improvements (Feb 2023)
    - 1.2.0: GS reconstruction implementation (Mar 2023)
    - 1.2.1: Fixed trial y-axis generation**: trial y-axes cover the full -90° to 90° range instead of only 0° to 90°** (Mar 2023)
    - 1.2.2: Reconstruction bug fixes (Mar 2023)
    - 1.3.0: Performance optimization and additional configurability in axis determination (Jul 2023)
    - 1.3.1: Improved performance using pinned memory (Jul 2023)
    - 1.3.2: Fixed the HT frame calculation and added an error check (Jul 2023)
    - 1.3.3: Option to limit the processing resolution, downsampling long windows to reduce memory use (Jul 2023)
    - 1.3.4: Field-aligned-flow support (Chen+22) (Jul 2023)
    - 1.3.5: Minor bug fix: Rfit is always computed, even when no threshold is set (Jul 2023)
    - 1.4.0: Improved determination of the vertical (y) axis**, using a trapezoid-rule mean field consistent with how A is integrated** (Jul 2023)
    - 1.4.1: Minor bug fix: fixed cross-product dimensions and the Alfvén velocity shape (Jul 2023)
    - 1.5.0: Massive performance improvement: data is downsampled once per window length before the sliding windows are generated, and windows are filtered in stages (Jul 2023)
    - 1.5.1: Use scatter instead of put, fixing an indexing bug when updating the y-axes of windows (Jul 2023)
    - 1.5.2: Use least squares instead of an exact solve for the HT frame, and a scatter bug fix (Jul 2023)
- Version 2: GS detection with reconstruction incorporated into the detection process
    - 2.0.0: Added GS reconstruction to the detection process (Sept 2023)
    - 2.1.0: Bundled `torchinterp1d` (Nov 2023)
- Version 3: Multi-spacecraft detection
    - 3.0.0: Clean up project structure
