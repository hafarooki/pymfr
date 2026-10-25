import numpy as np
from pymfr.analytic_models import gold_hoyle
from pytest import approx

def test_gold_hoyle():
    x = 50
    y = 20
    B0 = 10
    T0 = 1.5
    r = np.sqrt(x**2 + y**2)
    Bphi = B0 * T0 * r / (1 + T0**2 * r**2)
    angle = np.arctan2(y, x)
    Bx = -np.sin(angle) * Bphi
    By = np.cos(angle) * Bphi
    Bz = B0 / (1 + T0**2 * r**2)
    assert gold_hoyle(x=x, y=y, B0=B0, T0=T0) == approx((Bx, By, Bz))