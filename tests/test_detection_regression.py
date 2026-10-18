import pathlib

import numpy as np
import pytest
import scipy.constants
import torch

from pymfr.detect import detect_flux_ropes

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
INPUT_DATA_PATH = DATA_FOLDER_PATH / "wind_2016-01_1min_nonlin.npz"
# delete the CSV to regenerate the reference after an intended change
EXPECTED = DATA_FOLDER_PATH / "wind_2016-01_events.csv"


def test_detection_regression():
    data = np.load(INPUT_DATA_PATH)
    wind = {
        k: torch.as_tensor(data[k], dtype=torch.float32)
        for k in ("magnetic_field", "velocity", "density", "temperature")
    }
    window_lengths = np.unique(np.geomspace(10, 24 * 60, 200).astype(int))
    gas_pressure = (
        scipy.constants.Boltzmann
        * (wind["density"] * 1e6)
        * (wind["temperature"] * 1e6)
        * 1e9
    )
    results = detect_flux_ropes(
        wind["magnetic_field"],
        wind["velocity"],
        wind["density"],
        gas_pressure,
        batch_size=10000,
        window_steps=[1] * len(window_lengths),
        window_lengths=window_lengths,
        sample_spacing=60,
        n_trial_axes=256,
        threshold_diff=0.3,
        threshold_fit=0.3,
        max_processing_resolution=32,
        progress_bar=False,
    )
    start = results.window_start.values
    actual = np.column_stack(
        [
            start,
            start + results.window_length.values,
            results.axis_x,
            results.axis_y,
            results.axis_z,
        ]
    )
    actual = actual[np.lexsort((actual[:, 1], actual[:, 0]))]

    if not EXPECTED.exists():
        np.savetxt(
            EXPECTED,
            actual,
            fmt=["%d", "%d", "%.6f", "%.6f", "%.6f"],
            delimiter=",",
            header="start,end,axis_x,axis_y,axis_z",
        )
        pytest.skip(f"wrote reference events to {EXPECTED}")

    expected = np.loadtxt(EXPECTED, delimiter=",")
    assert len(actual) == len(expected)
    for (start, end, *axis), (expected_start, expected_end, *expected_axis) in zip(
        actual, expected
    ):
        assert (start, end) == (expected_start, expected_end)
        assert axis == pytest.approx(expected_axis, abs=1e-4)
