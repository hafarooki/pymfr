import numpy as np


def gold_hoyle(x, y, B0, T0):
    """
    Evaluate the Gold-Hoyle distribution at (x, y) [length units] assuming an
    axial field of B0 [arbitrary units] and angular twist of
    T0 [radians/length]. The coordinates are centered such that x = y = 0 is
    the center of the flux rope cross section.

    The model is given by:

        r = np.sqrt(x**2 + y**2)
        Bphi = B0 * T0 * r / (1 + T0**2 * r**2)
        angle = np.arctan2(y, x)
        Bx = -np.sin(angle) * Bphi
        Bx = np.cos(angle) * Bphi
        Bz = B0 / (1 + T0**2 * r**2)

    In Cartesian coordinates, sin(angle) = y/r and cos(angle) = x/r, so this
    simplifies to:
        denom = 1 + T0**2 * (x**2 + y**2)
        Bz = B0 / denom
        Bx = -B0 * T0 * y / denom
        By = B0 * T0 * x / denom

    Returns (Bx, By, Bz).
    """
    denom = 1 + T0**2 * (x**2 + y**2)
    Bz = B0 / denom
    Bx = -B0 * T0 * y / denom
    By = B0 * T0 * x / denom
    return Bx, By, Bz
