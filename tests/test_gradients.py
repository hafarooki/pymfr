import numpy as np
import pytest
import torch
from pytest import approx

import pymfr.gradients


@pytest.mark.parametrize("to_array", [np.asarray, torch.from_numpy], ids=["numpy", "torch"])
def test_minrmsdd(to_array):
    """Test that the expected zero-gradient direction is recovered."""
    # gradient for each component, chosen so that d/dy = 0
    dBx_dr = np.asarray([5 / 10, 0, 2 / 10])
    dBy_dr = np.asarray([-5 / 10, 0, -20 / 10])
    dBz_dr = np.asarray([3 / 10, 0, 4 / 10])

    # combine so that rows are magnetic field components
    gradient = np.stack([
        dBx_dr,
        dBy_dr,
        dBz_dr,
    ])

    gradients = np.repeat(gradient[None, :, :], 10, axis=0)

    result = pymfr.gradients.minrmsdd(to_array(gradients))
    # output should stay in the input's array library
    assert type(result) is type(to_array(gradients))
    result = np.asarray(result)

    # first, ensure it's a unit vector
    assert np.linalg.norm(result) == approx(1)

    assert abs(result[1]) == approx(1)


@pytest.mark.parametrize("to_array", [np.asarray, torch.from_numpy], ids=["numpy", "torch"])
def test_minrmsdd_batched_matches_individual(to_array):
    """Test that minrms(gradients)[i] == minrms(gradients[i]) up to a sign."""
    rng = np.random.default_rng(0)
    # two batch dims (2, 4), N=20 gradients averaged per batch, each 3x3
    gradients = rng.normal(size=(2, 4, 20, 3, 3))

    batched_result = np.asarray(pymfr.gradients.minrmsdd(to_array(gradients)))

    individuals_result = np.stack([
        np.stack([
            np.asarray(pymfr.gradients.minrmsdd(to_array(gradients[i, j])))
            for j in range(gradients.shape[1])
        ])
        for i in range(gradients.shape[0])
    ])

    assert batched_result.shape == individuals_result.shape == (2, 4, 3)

    # compare |dot(n_batched, n_individual)| to avoid sign ambiguity
    dot_product = np.sum(batched_result * individuals_result, axis=-1)
    assert np.abs(dot_product) == approx(np.ones((2, 4)))
