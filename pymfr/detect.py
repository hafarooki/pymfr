import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import xarray
from tensordict import TensorDict

from pymfr.gradients import minrmsdd
from pymfr.reconstruct import reconstruct_map
from pymfr.residue import (
    _calculate_folding_differences,
    _calculate_residue_diff,
)


def detect_flux_ropes(
    magnetic_field,
    velocity,
    density,
    gas_pressure,
    batch_size,
    window_lengths,
    window_steps,
    sample_spacing,
    magnetic_field_gradient=None,
    threshold_frame_quality=0.95,
    threshold_diff=0.3,
    threshold_fit=0.3,
    threshold_walen_slope=0.3,
    threshold_flow_field_alignment=0.8,
    n_trial_axes=256,
    max_processing_resolution=32,
    map_resolution=11,
    device=None,
    progress_bar=True,
) -> xarray.Dataset:
    """
    MFR detection based on the Grad-Shafranov automated detection algorithm.

    Parameters
    ----------
    magnetic_field : array_like, shape (N, 3)
        Magnetic field vectors in nT.
    velocity : array_like, shape (N, 3)
        Proton velocity vectors in km/s.
    density : array_like, shape (N,)
        Proton number density in cm^-3.
    gas_pressure : array_like, shape (N,)
        Plasma pressure in nPa.
    batch_size : int
        Number of windows to process per batch.
    window_lengths : sequence of int
        A range of sliding window sizes.
    window_steps : sequence of int
        A range of sliding window strides.
    sample_spacing : float
        Spacing between samples in seconds.
    magnetic_field_gradient : array_like, shape (N, 3, 3), optional
        Magnetic field gradient tensors in nT/km. If not given, the output
        ``minrmsdd`` is NaN.
    threshold_frame_quality : float, default 0.95
        Minimum correlation coefficient for the HT frame (Khabrov and Sonnerup,
        1998).
    threshold_diff : float, default 0.3
        The maximum allowable R_diff.
    threshold_fit : float, default 0.3
        The maximum allowable R_fit.
    threshold_walen_slope : float, default 0.3
        The threshold for <M_A> (calculated using Walen slope) to determine if the
        remaining flow is significant.
    threshold_flow_field_alignment : float or None, default 0.8
        The threshold for the correlation between v and v_A (validates that v is
        field aligned with constant M_A). If flux ropes with field aligned flows are
        not desired, then set to None.
    n_trial_axes : int, default 256
        The number of evenly spaced guess orientations used for the axis
        determination algorithm. Odd number is desirable so that one of the trial
        axes includes the previous iteration's best orientation (starting with avg
        field direction).
    max_processing_resolution : int or None, default 32
        Scale sliding window data down to approximately this size (if larger)
        before generating sliding windows to reduce memory usage. If None, windows
        are processed at full resolution.
    map_resolution : int, default 11
        Width and height in pixels of the reconstructed cross section maps.
        This determines the resolution used for GS reconstruction and the size
        of the ``y`` and ``x`` dims of the output ``map_*`` variables.
        It is recommended to leave this at its default value, which was chosen
        carefully after extensive analysis of many example events.
        A value too low would have too little detail and excessive smoothing,
        whereas a value too high can result in an unstable and meaningless reconstruction.
    device : str, torch.device, or None, default None
        Device to run on. None uses CUDA if available, then MPS, then CPU.
    progress_bar : bool, default True
        Whether to show a progress bar over window lengths.

    Returns
    -------
    xarray.Dataset
        Properties of the detected flux ropes, one entry along ``event`` each
        (zero entries if none were found). See the attributes for more details.

    References
    ----------
    Q. Hu, J. Zheng, Y. Chen, J. le Roux and L. Zhao, 2018, Automated detection of
    small-scale magnetic flux ropes in the solar wind: First results from the Wind
    spacecraft measurements, Astrophys. J. Supp., 239, 12,
    http://iopscience.iop.org/article/10.3847/1538-4365/aae57d
    """

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # convert all arrays to tensor so that numpy arrays can be given
    data = (
        TensorDict(
            magnetic_field=magnetic_field,
            velocity=velocity,
            density=density,
            gas_pressure=gas_pressure,
            batch_size=[len(magnetic_field)],
        )
        .float()  # convert to float32 explicitly also, for consistent behavior regardless of input dtype
        .to(device)
    )

    assert data["magnetic_field"].shape[1:] == (3,)
    assert data["velocity"].shape == data["magnetic_field"].shape
    assert data["density"].shape == data.batch_size
    assert data["gas_pressure"].shape == data.batch_size
    if magnetic_field_gradient is not None:
        data["magnetic_field_gradient"] = torch.as_tensor(
            magnetic_field_gradient, device=data.device, dtype=data.dtype
        )
        assert data["magnetic_field_gradient"].shape == data.batch_size + (3, 3)

    assert not any(torch.isnan(field).any() for field in data.values()), (
        "Data contains missing values (nan). PyMFR does not automatically handle missing values. Instead, please keep track of which data points contain nans beforehand, interpolate or fill, and then remove flux ropes with intervals containing too many NaNs."
    )

    # contains_existing keeps track of samples that were already confirmed as MFRs
    # for longer window lengths so that shorter windows do not attempt to overwrite them
    # initialized to all 0
    contains_existing = torch.zeros(
        len(magnetic_field), dtype=torch.bool, device=data.device
    )

    # accumulates across window sizes: one for each window size, then combined
    output_datasets = []

    assert len(set(window_lengths)) == len(window_lengths), (
        "window lengths must be unique"
    )
    # use sliding windows sorted by longest window length to shortest
    # because the algorithm gives preference to longer MFRs candidates over shorter ones
    # since long MFRs tend to show up as a bunch of small MFRs (especially ICMEs)
    for nominal_window_length, nominal_window_step in tqdm.tqdm(
        sorted(zip(window_lengths, window_steps, strict=True), reverse=True),
        disable=not progress_bar,
    ):
        output = _detect_for_window_length(
            data=data,
            contains_existing=contains_existing,
            nominal_window_length=nominal_window_length,
            nominal_window_step=nominal_window_step,
            batch_size=batch_size,
            sample_spacing=sample_spacing,
            threshold_frame_quality=threshold_frame_quality,
            threshold_diff=threshold_diff,
            threshold_fit=threshold_fit,
            threshold_walen_slope=threshold_walen_slope,
            threshold_flow_field_alignment=threshold_flow_field_alignment,
            n_trial_axes=n_trial_axes,
            max_processing_resolution=max_processing_resolution,
            map_resolution=map_resolution,
        )

        output_datasets.append(output)

    return xarray.concat(output_datasets, dim="event")


def _detect_for_window_length(
    data,
    contains_existing,
    nominal_window_length,
    nominal_window_step,
    batch_size,
    sample_spacing,
    threshold_frame_quality,
    threshold_diff,
    threshold_fit,
    threshold_walen_slope,
    threshold_flow_field_alignment,
    n_trial_axes,
    max_processing_resolution,
    map_resolution,
):
    # detect flux ropes using sliding windows of a single nominal length
    # bundle these together so they can be downsampled and windowed together
    original_timeseries = TensorDict(
        data=data,
        smooth_angles=_calculate_smooth_angles(data, nominal_window_length),
        overlaps=contains_existing.to(data.dtype),
        batch_size=data.batch_size,
        device=data.device,
    )
    scaled, window_length, window_step, windows = _downsample_and_window(
        original_timeseries,
        nominal_window_length,
        nominal_window_step,
        max_processing_resolution,
    )

    # one row per candidate window. each batch processing stage can drop
    # some rows if they need to be filtered out.
    # the results of each stage are added as new fields.
    # to avoid loading all of the windows into memory, this dict
    # contains the indices of the windowed samples.
    remaining_sliding_windows = TensorDict(
        window_index=torch.arange(len(windows), device=windows["start"].device),
        window_start=windows["start"],
        batch_size=windows.batch_size,
        device=data.device,
    )

    # estimate the HT frames
    remaining_sliding_windows["frame"] = _sliding_frames(
        scaled["data"], window_length, window_step
    )

    # first, filter out windows
    # where there is overlap with already-detected events,
    # or the magnetic field is not smooth enough
    window_overlaps = windows["samples", "overlaps"].amax(dim=1) > 0
    window_smoothness = windows["samples", "smooth_angles"].mean(dim=1)
    remaining_sliding_windows = remaining_sliding_windows[
        (~window_overlaps) & (window_smoothness >= 0.98)
    ]

    # stages of expensive batch processing,
    # which are performed on batches
    # to reduce memory usage. the later stages are more expensive,
    # hence this approach.
    stages = [
        partial(
            _find_and_check_vertical_directions,
            batch_size=batch_size,
            n_trial_axes=n_trial_axes,
        ),
        partial(
            _get_frame_quality_mask, threshold_frame_quality=threshold_frame_quality
        ),
        partial(
            _find_axes,
            threshold_diff=threshold_diff,
            threshold_walen_slope=threshold_walen_slope,
            threshold_flow_field_alignment=threshold_flow_field_alignment,
            n_trial_axes=n_trial_axes,
        ),
        partial(
            _validate_map,
            nominal_window_length=nominal_window_length,
            sample_spacing=sample_spacing,
            threshold_fit=threshold_fit,
            map_resolution=map_resolution,
        ),
    ]
    for stage in stages:
        passed = []
        for batch in remaining_sliding_windows.split(batch_size):
            mask, results = stage(
                batch, windows["samples", "data"][batch["window_index"]]
            )
            batch.update(results)
            passed.append(batch[mask])
        remaining_sliding_windows = torch.cat(passed)

    is_new = torch.zeros(
        len(remaining_sliding_windows), dtype=torch.bool, device=windows["start"].device
    )
    for i in torch.argsort(remaining_sliding_windows["residue_diff"], descending=False):
        start = remaining_sliding_windows["window_start"][i]

        if torch.any(contains_existing[start : start + nominal_window_length]):
            continue

        contains_existing[start : start + nominal_window_length] = True
        is_new[i] = True

    remaining_sliding_windows = remaining_sliding_windows[is_new]

    temporal_scale = (
        torch.ones_like(remaining_sliding_windows["window_start"], dtype=torch.float32)
        * nominal_window_length
        * sample_spacing
    )
    frames = remaining_sliding_windows["frame"]
    axes = remaining_sliding_windows["axis"]

    dx_dt = torch.linalg.norm(
        frames - (frames * axes).sum(dim=-1, keepdims=True) * axes,
        dim=-1,
    )
    remaining_sliding_windows["window_length"] = torch.full_like(
        remaining_sliding_windows["window_start"], nominal_window_length
    )
    remaining_sliding_windows["temporal_scale"] = temporal_scale
    remaining_sliding_windows["spatial_scale"] = dx_dt * temporal_scale

    return _to_dataset(remaining_sliding_windows)


def _to_dataset(windows):
    # convert the detected windows, a TensorDict with batch size (event,), into
    # the output dataset of detect_flux_ropes
    data = windows.cpu().numpy()
    data["velocity"] = data["frame"]

    variables = {
        "window_start": (
            ["event"],
            {
                "long_name": "Window start",
                "comment": "Index of the window's first sample in the input data",
            },
        ),
        "window_length": (
            ["event"],
            {
                "long_name": "Window length",
                "comment": "Approximate number of samples used from the input data",
            },
        ),
        "temporal_scale": (
            ["event"],
            {
                "long_name": "Duration",
                "comment": "Window duration",
                "units": "s",
            },
        ),
        "spatial_scale": (
            ["event"],
            {
                "long_name": "Size",
                "comment": "Distance traversed across the cross section",
                "units": "km",
            },
        ),
        "residue_diff": (
            ["event"],
            {
                "long_name": "R_diff",
                "comment": "RMS difference of the transverse pressure between "
                "the two branches of P_t(A), normalized by the range of P_t",
                "units": "1",
            },
        ),
        "residue_fit": (
            ["event"],
            {
                "long_name": "R_fit",
                "comment": "RMS residual of the polynomial fit of P_t(A) used for "
                "the reconstructed map, normalized by the range of the fit",
                "units": "1",
            },
        ),
        "walen_slope": (
            ["event"],
            {
                "long_name": "Walen slope",
                "comment": "Walen slope of the remaining flow in the HT frame",
                "units": "1",
            },
        ),
        "frame_quality": (
            ["event"],
            {
                "long_name": "HT frame quality",
                "comment": "HT frame correlation coefficient",
                "units": "1",
            },
        ),
        "flow_field_alignment": (
            ["event"],
            {
                "long_name": "Flow-field alignment",
                "comment": "Correlation between the HT-frame velocity and the "
                "magnetic field in Alfven units",
                "units": "1",
            },
        ),
        "axis": (
            ["event", "component"],
            {
                "long_name": "Axis",
                "comment": "Flux rope axis unit vector, in input coordinates",
                "units": "1",
            },
        ),
        "velocity": (
            ["event", "component"],
            {
                "long_name": "HT frame velocity",
                "comment": "HT frame velocity, in input coordinates",
                "units": "km/s",
            },
        ),
        "minrmsdd": (
            ["event", "component"],
            {
                "long_name": "MinRMSDD direction",
                "comment": "Minimum root-mean-square directional derivative "
                "unit vector, in input coordinates",
                "units": "1",
            },
        ),
        "map_Az": (
            ["event", "y", "x"],
            {
                "long_name": "A_z",
                "comment": "Magnetic flux function",
                "units": "nT km",
            },
        ),
        "map_B": (
            ["event", "y", "x", "component"],
            {
                "long_name": "B",
                "comment": "Magnetic field, in flux rope frame",
                "units": "nT",
            },
        ),
        "map_J": (
            ["event", "y", "x", "component"],
            {
                "long_name": "J",
                "comment": "Current density, in flux rope frame",
                "units": "pA/m^2",
            },
        ),
        "map_core_mask": (
            ["event", "y", "x"],
            {
                "long_name": "is_core",
                "comment": "Whether the pixel is inside the flux rope core "
                "(closed field lines)",
            },
        ),
    }

    return xarray.Dataset(
        data_vars={
            name: (dims, data[name], attrs) for name, (dims, attrs) in variables.items()
        },
    )


def _calculate_smooth_angles(data, window_length):
    magnetic_field = data["magnetic_field"]

    kernel_size = max(window_length // 10, 1)

    if kernel_size % 2 == 0:
        kernel_size += 1

    # simple moving average
    smoothed = F.avg_pool1d(
        magnetic_field.T,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        count_include_pad=False,
        stride=1,
    ).T

    return (F.normalize(smoothed, dim=1) * F.normalize(magnetic_field)).sum(dim=1)


def _downsample_and_window(
    original_timeseries,
    nominal_window_length,
    nominal_window_step,
    max_processing_resolution,
):
    """
    downsample timeseries so that a window of nominal_window_length
    samples spans at most max_processing_resolution samples, then split it
    into sliding windows.

    :param original_timeseries: TensorDict of fields with shape (time, ...), at full resolution.
    :param nominal_window_length: Window length in full resolution samples.
    :param nominal_window_step: Window step in full resolution samples.
    :param max_processing_resolution: Maximum window length after downsampling,
        or None to keep full resolution.
    :return: A tuple ``(series, window_length, window_step, windows)``:

        - ``series``: the downsampled series, shape (downsampled_time, ...).
        - ``window_length``: window length in downsampled samples.
        - ``window_step``: window step in downsampled samples.
        - ``windows``: TensorDict with batch size (window,) holding

          - ``start``: each window's start as a full resolution sample index.
          - ``samples``: lazy view of the series, shape
            (window, window_length, ...), where window_length is in
            downsampled samples. Indexing it with a tensor of window indices
            copies just those windows.
    """
    downsample_factor = (
        1
        if max_processing_resolution is None
        else max(nominal_window_length / max_processing_resolution, 1)
    )

    if downsample_factor > 1:
        # first, bring within a factor of 2 to max_processing_resolution
        kernel_size = math.floor(downsample_factor)

        # at this point, the window length is still the following factor above max processing resolution
        remaining_factor = (
            nominal_window_length / kernel_size
        ) / max_processing_resolution

        # apply downsample
        downsampled_timeseries = original_timeseries.apply(
            partial(
                _downsample, kernel_size=kernel_size, scale_factor=1 / remaining_factor
            ),
            batch_size=[],
        )

        # the new length isn't known before the op runs, so apply it
        # unbatched and then take the batch size from the first dim
        downsampled_timeseries = downsampled_timeseries.auto_batch_size_(batch_dims=1)
        window_length = max_processing_resolution
    else:
        downsampled_timeseries = original_timeseries
        window_length = nominal_window_length

    window_step = max(1, math.floor(nominal_window_step / downsample_factor))

    # window i covers downsampled samples i * window_step to
    # i * window_step + window_length
    n_window = (len(downsampled_timeseries) - window_length) // window_step + 1
    # full-resolution sample indices
    window_starts = (
        torch.arange(n_window, device=downsampled_timeseries.device)
        * window_step
        * downsample_factor
    ).to(int)

    # generate the sliding windows as a tensor view, which is only actually loaded into memory when copied on demand
    window_samples = downsampled_timeseries.apply(
        lambda field: (
            field
            # puts each window's samples last: (window, ..., size)
            .unfold(0, window_length, window_step)
            # move them next to the window axis: (window, size, ...)
            .movedim(-1, 1)
        ),
        batch_size=[n_window, window_length],
    )
    windows = TensorDict(
        start=window_starts,
        samples=window_samples,
        batch_size=[n_window],
    )

    return downsampled_timeseries, window_length, window_step, windows


def _downsample(field: torch.Tensor, kernel_size, scale_factor) -> torch.Tensor:
    """
    average field, shape (time, ...), over blocks of kernel_size samples, then
    linearly resample the result by scale_factor
    """
    # torch 1D operations require input of shape (batch, channel, time)
    n_time = field.shape[0]
    # (time,) or (time, 3) -> (1, channel, time)
    channels_first = field.reshape(n_time, -1).T.unsqueeze(0)

    pooled = F.avg_pool1d(channels_first, kernel_size, kernel_size)

    # interpolate to bring window length down to max_processing_resolution
    resampled = F.interpolate(
        pooled,
        scale_factor=scale_factor,
        mode="linear",
        align_corners=True,
    )

    # (1, channel, new_time) -> (new_time, channel)
    time_first = resampled.squeeze(0).T
    # -> (new_time,) or (new_time, 3)
    return time_first.reshape(time_first.shape[0], *field.shape[1:])


def _sliding_frames(data, window_length, window_step):
    magnetic_field = data["magnetic_field"]
    velocity = data["velocity"]

    electric_field = -torch.cross(velocity, magnetic_field, dim=-1)

    Bx, By, Bz = magnetic_field[..., 0], magnetic_field[..., 1], magnetic_field[..., 2]
    coefficients = [
        [By**2 + Bz**2, -Bx * By, -Bx * Bz],
        [-Bx * By, Bx**2 + Bz**2, -By * Bz],
        [-Bx * Bz, -By * Bz, Bx**2 + By**2],
    ]
    coefficient_matrix = torch.zeros(
        (
            *magnetic_field.shape[:-2],
            math.floor((magnetic_field.shape[-2] - window_length) / window_step + 1),
            3,
            3,
        ),
        device=data.device,
        dtype=magnetic_field.dtype,
    )
    for i in range(3):
        for j in range(3):
            coefficient_matrix[..., i, j] = F.avg_pool1d(
                coefficients[i][j].unsqueeze(-2), window_length, window_step
            ).squeeze(-2)

    dependent_values = torch.cross(electric_field, magnetic_field, dim=-1)
    dependent_values = F.avg_pool1d(dependent_values.T, window_length, window_step).T

    try:
        frames = torch.linalg.lstsq(coefficient_matrix, dependent_values).solution
    except RuntimeError:
        # lstsq appears to be buggy on GPU so do it on CPU if it fails
        frames = torch.linalg.lstsq(
            coefficient_matrix.cpu(), dependent_values.cpu()
        ).solution.to(data.device)

    return frames


def _find_and_check_vertical_directions(batch, batch_data, batch_size, n_trial_axes):
    """
    estimate the flux rope y-axis of each window, and keep only the windows
    where the field component along it changes sign exactly once
    """
    batch_magnetic_field = batch_data["magnetic_field"]
    window_frames = batch["frame"]

    if len(batch) == 0:
        # avg_pool1d rejects an empty batch, so return empty results
        return torch.zeros(0, dtype=torch.bool, device=batch.device), {
            "vertical_direction": torch.zeros_like(window_frames)
        }

    window_avg_field_strength = torch.norm(batch_magnetic_field, dim=-1).mean(dim=-1)

    # trapezoid-integral mean of each window's field
    # L samples -> L - 1 gaps
    window_avg_field = torch.trapezoid(batch_magnetic_field, dim=-2) / (
        batch_magnetic_field.shape[-2] - 1
    )

    avg_field_direction = F.normalize(window_avg_field, dim=-1)
    path_direction = -F.normalize(window_frames, dim=-1)
    cross_product = torch.cross(avg_field_direction, path_direction, dim=-1)
    vertical_directions = F.normalize(cross_product, dim=-1)

    # compare ratio of average along guessed vertical direction and the one perpendicular. if less than 10x difference, not well determined
    perpendicular_direction = F.normalize(
        torch.cross(path_direction, vertical_directions, dim=-1), dim=-1
    )
    ratio = (window_avg_field * perpendicular_direction).sum(
        dim=-1
    ) / window_avg_field_strength
    too_close = (ratio < 0.1) | (
        torch.norm(vertical_directions, dim=-1) == 0
    )  # also too close if norm of vertical direction is 0

    too_close_indices = torch.nonzero(too_close).flatten()

    if len(too_close_indices) > 0:
        too_close_field = batch_magnetic_field[too_close_indices]
        too_close_mean_directions = avg_field_direction.index_select(
            0, too_close_indices
        )
        too_close_frames = window_frames.index_select(0, too_close_indices)
        too_close_path_directions = path_direction.index_select(0, too_close_indices)

        # basis vectors will be the ones with the lowest component value in too_close_path_direction,
        # effectively the one with the lowest dot product and thus most perpendicular
        basis_vectors = torch.zeros_like(too_close_path_directions)
        basis_vectors.scatter_(
            dim=1,
            index=too_close_path_directions.abs().argmin(dim=-1, keepdim=True),
            value=1,
        )

        # guaranteed perpendicular to too_close_path_directions and nonzero as long as velocity is nonzero
        arbitrary_perpendicular_directions = F.normalize(
            torch.cross(basis_vectors, too_close_path_directions, dim=-1), dim=-1
        )

        # to be used for linear combinations for vertical trials
        alternative_directions = F.normalize(
            torch.cross(
                arbitrary_perpendicular_directions, too_close_path_directions, dim=-1
            ),
            dim=-1,
        )

        n_vertical_trials = 180

        angle = torch.linspace(
            -np.pi / 2,
            np.pi / 2,
            n_vertical_trials,
            device=arbitrary_perpendicular_directions.device,
        ).reshape(1, n_vertical_trials, 1)
        possible_vertical_directions = F.normalize(
            arbitrary_perpendicular_directions.unsqueeze(1) * torch.cos(angle)
            + alternative_directions.unsqueeze(1) * torch.sin(angle),
            dim=-1,
        )

        # ensure that x unit points opposite direction of frames
        x_unit = F.normalize(
            torch.cross(
                possible_vertical_directions,
                too_close_mean_directions.unsqueeze(1),
                dim=-1,
            ),
            dim=-1,
        )
        x_unit_sign = -torch.sign(
            (x_unit * too_close_frames.unsqueeze(1)).sum(dim=-1, keepdim=True)
        )
        possible_vertical_directions *= x_unit_sign

        n_too_close = len(too_close_indices)

        # number processed = n in vertical axis batch * n too close
        # -> n in axis batch = number processed / n too close
        # max number processed (ideal for batch processing) would be
        # total number of trial vertical axes * n too close
        # but it cant be more than batch size
        # also must have minimum axis batch size 1 to get anywhere
        # hence the following line of code makes sense
        axis_batch_step = max(
            1, min(n_vertical_trials * n_too_close, batch_size) // n_too_close
        )

        minimum_residues = []  # best residue attainable for each vertical axis

        for axis_batch_directions in possible_vertical_directions.split(
            axis_batch_step, dim=1
        ):
            # broadcast view pairing each field with each trial direction
            axis_batch_field = too_close_field.unsqueeze(1).expand(
                -1, axis_batch_directions.shape[1], -1, -1
            )
            axis_batch_potential = _calculate_normalized_potential(
                axis_batch_field, axis_batch_directions
            )
            axis_batch_inflection_mask = (
                _is_good_vertical_component(axis_batch_field, axis_batch_directions)
            ) & (
                torch.abs(axis_batch_potential[..., -1])
                < torch.abs(axis_batch_potential).amax(dim=-1) * 0.1
            )
            axis_batch_min_residue = torch.full(
                axis_batch_inflection_mask.shape,
                torch.inf,
                dtype=axis_batch_directions.dtype,
                device=axis_batch_directions.device,
            )

            if torch.any(axis_batch_inflection_mask):
                axis_batch_min_residue[axis_batch_inflection_mask] = (
                    _minimize_axis_residue(
                        axis_batch_field[axis_batch_inflection_mask],
                        axis_batch_directions[axis_batch_inflection_mask],
                        axis_batch_potential[axis_batch_inflection_mask],
                        n_trial_axes,
                    )[1]
                )

            minimum_residues.append(axis_batch_min_residue)

        # looks like [[residue1a, ..., residue1z], [residue2a, ..., residue2z]]
        min_residue = torch.concat(minimum_residues, dim=1)
        assert min_residue.shape == (n_too_close, n_vertical_trials)
        argmin = min_residue.argmin(dim=-1)
        best = torch.take_along_dim(
            possible_vertical_directions, argmin[:, None, None], dim=1
        ).squeeze(1)
        vertical_directions[too_close_indices] = best

    return _is_good_vertical_component(batch_magnetic_field, vertical_directions), {
        "vertical_direction": vertical_directions
    }


def _calculate_normalized_potential(magnetic_field, vertical_directions):
    # calculate the A array/magnetic flux function/potential using $A(x) = -\int_{x_0}^x B_y dx$
    batch_magnetic_field_y = (
        magnetic_field @ vertical_directions.unsqueeze(-1)
    ).squeeze(-1)
    unscaled_potential = torch.zeros(
        batch_magnetic_field_y.shape, device=batch_magnetic_field_y.device
    )
    unscaled_potential[..., 1:] = torch.cumulative_trapezoid(batch_magnetic_field_y)

    # use the potential peaks as the inflection points
    inflection_points = unscaled_potential[..., 1:-1].abs().argmax(dim=-1) + 1

    peak_values = unscaled_potential.gather(-1, inflection_points.unsqueeze(-1))
    normalized_potential = unscaled_potential / peak_values

    return normalized_potential


def _is_good_vertical_component(magnetic_field, vertical_directions):
    magnetic_field_y = torch.einsum(
        "...tc,...c->...t", magnetic_field, vertical_directions
    )

    kernel_size = max(magnetic_field_y.shape[-1] // 10, 1)

    if kernel_size % 2 == 0:
        kernel_size += 1

    # simple moving average
    smoothed = F.avg_pool1d(
        magnetic_field_y,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        count_include_pad=False,
        stride=1,
    )

    is_sign_change = (torch.diff(torch.sign(smoothed)) != 0).to(int)

    single_stationary_point = torch.sum(is_sign_change, dim=-1) == 1

    return single_stationary_point


def _minimize_axis_residue(
    magnetic_field, vertical_directions, potential, n_trial_axes, return_residue=True
):
    # use avg magnetic field as initial guess
    initial_guess = magnetic_field.mean(dim=-2)
    # ensure perpendicular to vertical direction
    initial_guess = (
        initial_guess
        - (initial_guess * vertical_directions).sum(dim=-1, keepdims=True)
        * vertical_directions
    )
    initial_axes = F.normalize(initial_guess, dim=-1)
    # generate trial axes by rotating <B> around the vertical axis (we expect z to be <B> +- 90 degrees around y, since <Bz> should be positive)
    alternative_directions = F.normalize(
        torch.cross(vertical_directions, initial_axes, dim=-1), dim=-1
    )
    angle = torch.linspace(
        -np.pi / 2, np.pi / 2, n_trial_axes, device=initial_axes.device
    ).reshape(1, n_trial_axes, 1)
    trial_axes = F.normalize(
        initial_axes.unsqueeze(1) * torch.cos(angle)
        + alternative_directions.unsqueeze(1) * torch.sin(angle),
        dim=-1,
    )

    # For determining the axis we only use magnetic field data, not gas pressure
    # This is because gas pressure error diff is not affected by rotations about the y axis
    # unlike Bz, which depends on the axis orientation
    # Bz itself is conserved along field lines, and this way we can compare the SIGN of Bz as well
    error_diff_axis = _calculate_residue_map(magnetic_field, trial_axes, potential)

    # select the best orientation
    residue, argmin = error_diff_axis.min(dim=-1)
    index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
    optimal_axes = trial_axes.gather(-2, index).squeeze(-2)

    if return_residue:
        return optimal_axes, residue
    else:
        return optimal_axes


def _calculate_residue_map(magnetic_field, trial_axes, potential):
    difference_vectors = torch.stack(
        [
            _calculate_folding_differences(potential, magnetic_field[:, :, i])
            for i in range(3)
        ],
        dim=-1,
    )

    differences = torch.einsum("bsc,bkc->bsk", difference_vectors, trial_axes)
    axial = torch.einsum("btc,bkc->btk", magnetic_field, trial_axes)

    min_axial, max_axial = torch.aminmax(axial, dim=1)
    axial_range = max_axial - min_axial
    error_diff = torch.mean(differences**2, dim=1) ** 0.5 / axial_range

    return error_diff


def _get_frame_quality_mask(batch, batch_data, threshold_frame_quality):
    batch_frames = batch["frame"]
    batch_magnetic_field = batch_data["magnetic_field"]
    batch_velocity = batch_data["velocity"]
    batch_electric_field = -torch.cross(batch_velocity, batch_magnetic_field, dim=-1)

    frame_quality = _calculate_frame_quality(
        batch_frames, batch_magnetic_field, batch_velocity, batch_electric_field
    )
    return frame_quality > threshold_frame_quality, {"frame_quality": frame_quality}


def _calculate_frame_quality(frames, magnetic_field, velocity, electric_field):
    electric_field = -torch.cross(velocity, magnetic_field, dim=-1)

    remaining_electric_field = -torch.cross(
        velocity - frames.unsqueeze(1), magnetic_field, dim=2
    )

    frame_error = (remaining_electric_field**2).sum(dim=2).mean(dim=1) / (
        electric_field**2
    ).sum(dim=2).mean(dim=1)
    frame_quality = torch.sqrt(1 - frame_error)

    return frame_quality


def _find_axes(
    batch,
    batch_data,
    threshold_walen_slope,
    threshold_flow_field_alignment,
    threshold_diff,
    n_trial_axes,
):
    batch_normalized_potential = _calculate_normalized_potential(
        batch_data["magnetic_field"], batch["vertical_direction"]
    )

    batch_axes, axis_residue = _minimize_axis_residue(
        batch_data["magnetic_field"],
        batch["vertical_direction"],
        batch_normalized_potential,
        n_trial_axes,
    )

    axial_component = (batch_data["magnetic_field"] * batch_axes[:, None]).sum(-1)

    walen_slope, flow_field_alignment = _calculate_walen_slope(
        batch["frame"],
        batch_data["magnetic_field"],
        batch_data["density"],
        batch_data["velocity"],
    )
    walen_slope_mask = torch.abs(walen_slope) <= threshold_walen_slope

    if threshold_flow_field_alignment is not None:
        walen_slope_mask = walen_slope_mask | (
            (torch.abs(flow_field_alignment) >= threshold_flow_field_alignment)
            & (torch.abs(walen_slope) < 0.9)
        )

    alpha = torch.where(
        torch.abs(walen_slope) < threshold_walen_slope, 0, walen_slope**2
    ).unsqueeze(1)

    transverse_pressure = _calculate_transverse_pressure(
        torch.linalg.norm(batch_data["magnetic_field"], dim=-1),
        axial_component,
        batch_data["gas_pressure"],
        alpha,
    )

    error_diff = _calculate_residue_diff(
        batch_normalized_potential, transverse_pressure
    )
    Bz_fluctuation = axial_component - axial_component.mean(dim=-1, keepdims=True)
    pressure_fluctuation = transverse_pressure - transverse_pressure.mean(
        dim=-1, keepdims=True
    )
    potential_fluctuation = (
        batch_normalized_potential
        - batch_normalized_potential.mean(dim=-1, keepdims=True)
    )
    Bz_sign = torch.sign(
        (Bz_fluctuation * potential_fluctuation).sum(dim=-1)
        / (potential_fluctuation**2).sum(dim=-1)
    )
    transverse_pressure_sign = torch.sign(
        (pressure_fluctuation * potential_fluctuation).sum(dim=-1)
        / (potential_fluctuation**2).sum(dim=-1)
    )

    # combine all the masks
    mask = (
        walen_slope_mask
        & (axis_residue <= threshold_diff)
        & (error_diff <= threshold_diff)
        & (transverse_pressure_sign == 1)
        & (Bz_sign == 1)
    )

    if "magnetic_field_gradient" in batch_data.keys():  # noqa: SIM118
        # Claude: eigh isn't implemented on MPS, so compute it on CPU there
        gradient = batch_data["magnetic_field_gradient"]
        if gradient.device.type == "mps":
            batch_minrmsdd = minrmsdd(gradient.cpu()).to(gradient.device)
        else:
            batch_minrmsdd = minrmsdd(gradient)
    else:
        batch_minrmsdd = torch.full_like(batch_axes, torch.nan)

    return mask, {
        "axis": batch_axes,
        "minrmsdd": batch_minrmsdd,
        "residue_diff": error_diff,
        "walen_slope": walen_slope,
        "flow_field_alignment": flow_field_alignment,
    }


def _calculate_walen_slope(batch_frames, magnetic_field, density, velocity):
    # alfven velocity is calculated based on v_A = |B|/sqrt(n_p m_p mu_0)
    # constant factor 21.8114 assumes B in nT, n_p in cm^-3, v_A in km/s
    alfven_velocity = (magnetic_field / torch.sqrt(density.unsqueeze(-1))) * 21.8114
    alfven_velocity = torch.where(density[:, :, None] == 0, torch.inf, alfven_velocity)

    remaining_flow = velocity - batch_frames.unsqueeze(1)

    # simple linear regression slope with y intercept fixed at 0
    walen_slope = (remaining_flow.flatten(1) * alfven_velocity.flatten(1)).sum(
        dim=1
    ) / (alfven_velocity.flatten(1) ** 2).sum(dim=1)

    field_alignment = (
        F.normalize(
            remaining_flow.flatten(1)
            - remaining_flow.flatten(1).mean(dim=1, keepdim=True),
            dim=1,
        )
        * F.normalize(
            alfven_velocity.flatten(1)
            - alfven_velocity.flatten(1).mean(dim=1, keepdim=True),
            dim=1,
        )
    ).sum(dim=1)

    return walen_slope, field_alignment


def _calculate_transverse_pressure(B_magnitude, B_transverse, Pgas, alpha):
    transverse_magnetic_pressure = (
        (B_transverse * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
    )
    isotropic_magnetic_pressure = (
        (B_magnitude * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
    )
    # calculate $p_t = B_z^2/2mu_0 + p/(1-alpha) + (alpha/(1-alpha)) B^2/2mu_0$
    # since alpha is constant, no need for extra factor of (1 - alpha) used in Chen et al
    # because \del^2 A' = \del^2 ((1 - alpha) A) = (1 - alpha) \del^2 A
    # and dPt'/dA' = (1/(1-alpha))dPt'/dA
    # also, \del^2 A = -\mu_0 j_z for any 2D field, field aligned flow or not,
    # so \del^2 A version of Pt is more useful than the one for \del^2 A'
    transverse_pressure = transverse_magnetic_pressure + (
        Pgas + alpha * isotropic_magnetic_pressure
    ) / (1 - alpha)
    return transverse_pressure


def _validate_map(
    batch,
    batch_data,
    nominal_window_length,
    sample_spacing,
    threshold_fit,
    map_resolution,
):
    if len(batch) == 0:
        # the reconstruction can't handle an empty batch, so explicitly define the empty-case result
        empty_map = torch.zeros(
            (0, map_resolution, map_resolution), device=batch.device
        )
        empty_vector_map = torch.zeros(
            (0, map_resolution, map_resolution, 3), device=batch.device
        )
        return torch.zeros(0, dtype=torch.bool, device=batch.device), {
            "map_Az": empty_map,
            "map_B": empty_vector_map,
            "map_J": empty_vector_map,
            "map_core_mask": empty_map.to(torch.bool),
            "residue_fit": torch.zeros(0, device=batch.device),
        }

    batch_axes = batch["axis"]
    batch_frames = batch["frame"]
    batch_walen_slope = batch["walen_slope"]

    reconstructed_map = _generate_map(
        batch_data,
        batch_axes,
        batch_frames,
        batch_walen_slope,
        nominal_window_length,
        sample_spacing,
        map_resolution,
    )

    y_observed = reconstructed_map.magnetic_potential.shape[-2] // 2

    sign = torch.sign(
        reconstructed_map.magnetic_potential[:, y_observed, :].mean(dim=1)
    )

    mask_fit = reconstructed_map.error_fit <= threshold_fit

    closed_values = torch.abs(
        torch.where(
            reconstructed_map.core_mask,
            reconstructed_map.magnetic_potential * sign[:, None, None],
            torch.nan,
        )
    )
    peak_closed = torch.nan_to_num(closed_values, -torch.inf).flatten(-2).max(dim=-1)
    peak_observed_value = torch.nan_to_num(closed_values, -torch.inf)[
        :, y_observed, :
    ].amax(dim=-1)
    minimum_closed = torch.nan_to_num(closed_values, torch.inf).flatten(-2).min(dim=-1)

    height = torch.any(reconstructed_map.core_mask, dim=-1).sum(dim=-1)
    width = torch.any(reconstructed_map.core_mask, dim=-2).sum(dim=-1)

    closed_current_values = torch.abs(
        torch.where(
            reconstructed_map.core_mask,
            reconstructed_map.current_density_z * sign[:, None, None],
            torch.nan,
        )
    )
    current_at_peak = (
        closed_current_values.flatten(-2)
        .gather(-1, peak_closed.indices.unsqueeze(-1))
        .squeeze(-1)
    )
    current_at_minimum = (
        closed_current_values.flatten(-2)
        .gather(-1, minimum_closed.indices.unsqueeze(-1))
        .squeeze(-1)
    )
    minimum_current = (
        torch.nan_to_num(closed_current_values, torch.inf).flatten(-2).amin(dim=-1)
    )
    maximum_current = (
        torch.nan_to_num(closed_current_values, -torch.inf).flatten(-2).amax(dim=-1)
    )

    mask_current = (current_at_minimum <= minimum_current) & (
        current_at_peak >= maximum_current
    )

    mask_closed = (
        (width >= 5) & (height >= 5) & (minimum_closed.values <= peak_observed_value)
    )

    mask_valid = ~torch.any(
        (
            torch.isnan(reconstructed_map.magnetic_potential)
            | torch.isinf(reconstructed_map.magnetic_potential)
        ).flatten(-2),
        dim=-1,
    )

    return mask_fit & mask_valid & mask_current & mask_closed, {
        "map_Az": reconstructed_map.magnetic_potential,
        "map_B": torch.stack(
            [
                reconstructed_map.magnetic_field_x,
                reconstructed_map.magnetic_field_y,
                reconstructed_map.magnetic_field_z,
            ],
            dim=-1,
        ),
        "map_J": torch.stack(
            [
                reconstructed_map.current_density_x,
                reconstructed_map.current_density_y,
                reconstructed_map.current_density_z,
            ],
            dim=-1,
        ),
        "map_core_mask": reconstructed_map.core_mask,
        "residue_fit": reconstructed_map.error_fit,
    }


def _generate_map(
    data,
    axes,
    frames,
    walen_slope,
    nominal_window_length,
    sample_spacing,
    map_resolution,
):
    batch_magnetic_field = data["magnetic_field"]
    batch_gas_pressure = data["gas_pressure"]

    alpha = walen_slope**2

    time_elapsed = nominal_window_length * sample_spacing
    dx_dt = torch.linalg.norm(
        frames - (frames * axes).sum(dim=-1, keepdims=True) * axes, dim=-1
    )

    dt = time_elapsed / batch_magnetic_field.shape[-2]
    dx = dx_dt * dt

    x_axis = F.normalize(
        -(frames - (frames * axes).sum(dim=-1, keepdim=True) * axes), dim=-1
    )
    y_axis = F.normalize(torch.cross(axes, x_axis, dim=-1), dim=-1)
    basis = torch.stack([x_axis, y_axis, axes], dim=-1)

    input_B = batch_magnetic_field @ basis

    reconstructed_map = reconstruct_map(
        input_B,
        batch_gas_pressure,
        alpha=alpha,
        sample_spacing=dx,
        poly_order=3,
        resolution=map_resolution,
        pad=0,
    )

    return reconstructed_map
