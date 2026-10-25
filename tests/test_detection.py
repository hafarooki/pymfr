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


@pytest.fixture
def sample_data():
    data = dict(np.load(INPUT_DATA_PATH))
    data["gas_pressure"] = (
        scipy.constants.Boltzmann
        * (data["density"] * 1e6)
        * (data["temperature"] * 1e6)
        * 1e9
    )
    return {
        k: torch.as_tensor(data[k], dtype=torch.float32)
        for k in ("magnetic_field", "velocity", "density", "gas_pressure")
    }

@pytest.fixture
def detection_kwargs():
    window_lengths = np.unique(np.geomspace(10, 24 * 60, 200).astype(int))
    return {
        "batch_size": 10000,
        "window_steps": [1] * len(window_lengths),
        "window_lengths": window_lengths,
        "sample_spacing": 60,
        "n_trial_axes": 256,
        "threshold_diff": 0.3,
        "threshold_fit": 0.3,
        "max_processing_resolution": 32,
        "progress_bar": False,
    }


def test_detection_regression(sample_data, detection_kwargs):
    results = detect_flux_ropes(
        **sample_data,
        **detection_kwargs,
    )
    start = results.window_start.values
    actual = np.column_stack(
        [
            start,
            start + results.window_length.values,
            results.axis.values,
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

def test_output_includes_minrmsdd_when_provided_gradient(sample_data, detection_kwargs):
    sample_data = {key: value[:24 * 60] for key, value in sample_data.items()}
    magnetic_field_gradient = np.zeros(
        shape=(len(sample_data["magnetic_field"]), 3, 3),
        dtype=float
    )
    results = detect_flux_ropes(
        **sample_data,
        **detection_kwargs,
        magnetic_field_gradient=magnetic_field_gradient
    )

    assert "minrmsdd" in results.data_vars
