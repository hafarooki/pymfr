import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm as tqdm
import math
from functools import partial
from torchinterp1d import interp1d
from xarray import Dataset
import xarray

from pymfr.residue import _calculate_residue_diff, _calculate_folding_differences, _calculate_interpolated_values
from pymfr.reconstruct import reconstruct_map


def detect_flux_ropes(magnetic_field,
                      velocity,
                      density,
                      gas_pressure,
                      batch_size,
                      window_lengths,
                      window_steps,
                      sample_spacing,
                      threshold_frame_quality=0.95,
                      threshold_diff=0.3,
                      threshold_fit=0.2,
                      threshold_walen_slope=0.3,
                      threshold_flow_field_alignment=0.8,
                      n_trial_axes=256,
                      max_processing_resolution=None,
                      cuda=True,
                      progress_bar=True):
    """
    MFR detection based on the Grad-Shafranov automated detection algorithm.

    References:
    Q. Hu, J. Zheng, Y. Chen, J. le Roux and L. Zhao, 2018,
    Automated detection of small-scale magnetic flux ropes in the solar wind:
    First results from the Wind spacecraft measurements, Astrophys. J. Supp., 239, 12,
    http://iopscience.iop.org/article/10.3847/1538-4365/aae57d

    :param magnetic_field: Magnetic field vectors in nT, shape (N, 3).
    :param velocity: Proton velocity vectors in km/s, shape (N, 3).
    :param density: Proton number density in cm^-3, shape (N,).
    :param batch_size_: Number of windows to process per batch.
    :param window_lengths: A range of sliding window sizes.
    :param window_steps: A range of sliding window strides.
    :param sample_spacing: Spacing between samples in seconds.
    :param threshold_frame_quality: Minimum correlation coefficient for the HT frame (Khabrov and Sonnerup, 1998). Default 0.95
    :param threshold_diff: The maximum allowable R_diff
    :param threshold_fit: The maximum allowable R_fit
    :param threshold_walen_slope: The threshold for <M_A> (calculated using Walen slope) to determine if the remaining flow is significant. Default 0.1.
    :param threshold_flow_field_alignment: The threshold for the correlation between v and v_A
    (validates that v is field aligned with constant M_A).
    If flux ropes with field aligned flows are not desired, then set to None.
    Default 0.8.
    :param n_trial_axes: The number of evenly spaced guess orientations used for the axis determination algorithm.
    Odd number is desirable so that one of the trial axes includes the previous iteration's best orientation (starting with avg field direction).
    :param max_processing_resolution: Scale sliding window data down to approximately this size (if larger) before generating sliding windows to reduce memory usage.
    :param cuda: Whether to use the GPU
    :return: A DataFrame describing the detected flux ropes.
    """

    full_tensor = _pack_data(magnetic_field, velocity, density, gas_pressure)

    if cuda:
        full_tensor = full_tensor.cuda()

    # these are updated as the algorithm runs
    # contains_existing keeps track of samples that were already confirmed as MFRs
    # for longer window lengths so that shorter windows do not attempt to overwrite them
    contains_existing = torch.zeros(len(magnetic_field), dtype=torch.bool, device=full_tensor.device)

    output_datasets = []

    sliding_windows = _get_sliding_windows(window_lengths, window_steps)

    for nominal_window_length, nominal_window_step in (
    sliding_windows if not progress_bar else tqdm.tqdm(sliding_windows)):
        scaled_tensor = full_tensor
        scaled_overlaps = contains_existing.to(float).unsqueeze(0)
        downsample_factor = 1 if max_processing_resolution is None else max(
            nominal_window_length / max_processing_resolution, 1)
        if downsample_factor > 1:
            # first, bring within a factor of 2 to max_processing_resolution
            scaled_tensor = F.avg_pool1d(full_tensor.T, math.floor(downsample_factor), math.floor(downsample_factor)).T

            # at this point, the window length is still the following factor above max processing resolution
            remaining_factor = (nominal_window_length / math.floor(downsample_factor)) / max_processing_resolution

            # interpolate to bring window length down to max_processing_resolution
            scaled_tensor = F.interpolate(scaled_tensor.T.unsqueeze(0), scale_factor=1 / remaining_factor,
                                          mode="linear", align_corners=True).squeeze(0).T

            # do the same for overlap flags, except maxpool instead of avgpool so that none of the points that are scaled down overlap
            scaled_overlaps = F.max_pool1d(scaled_overlaps, math.floor(downsample_factor),
                                           math.floor(downsample_factor))

            # interpolate so that if there is one neighboring that is positive, the interpolated value will be greater than 1
            scaled_overlaps = F.interpolate(scaled_overlaps.unsqueeze(0), scale_factor=1 / remaining_factor,
                                            mode="linear", align_corners=True).squeeze(0)

        window_length = nominal_window_length if max_processing_resolution is None else min(nominal_window_length,
                                                                                            max_processing_resolution)
        window_step = max(1, math.floor(nominal_window_step / downsample_factor))

        window_avg_field_strength = F.avg_pool1d(torch.norm(scaled_tensor[:, :3], dim=1).unsqueeze(0), window_length,
                                                 window_step).squeeze(0)
        window_overlaps = F.max_pool1d(scaled_overlaps, window_length, window_step).squeeze(0) > 0

        # generate the sliding windows as a tensor view, which is only actually loaded into memory when copied on demand
        window_data = scaled_tensor.unfold(size=window_length, step=window_step, dimension=0)
        window_starts = (
                    torch.arange(len(window_data), device=scaled_tensor.device) * window_step * downsample_factor).to(
            int)

        window_frames = _sliding_frames(scaled_tensor, window_length, window_step)

        window_vertical_directions = _sliding_vertical_directions(scaled_tensor, window_length, window_step,
                                                                  window_data,
                                                                  window_frames,
                                                                  window_avg_field_strength,
                                                                  batch_size,
                                                                  n_trial_axes)

        output = _batch_process(batch_size,
                                nominal_window_length,
                                sample_spacing,
                                threshold_frame_quality,
                                threshold_diff,
                                threshold_fit,
                                threshold_walen_slope,
                                threshold_flow_field_alignment,
                                n_trial_axes,
                                window_overlaps,
                                window_data,
                                window_frames,
                                window_starts,
                                window_vertical_directions,
                                contains_existing)

        if output is None:
            continue

        output_datasets.append(output)

    return xarray.concat(output_datasets, dim="event")


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, window_data, good_windows, batch_size):
        super(SlidingWindowDataset).__init__()
        self.window_data = window_data
        self.good_windows = good_windows
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.good_windows) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.good_windows[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.window_data.index_select(0, batch_indices).transpose(1, 2).contiguous()


def _batch_process(batch_size,
                   nominal_window_length,
                   sample_spacing,
                   threshold_frame_quality,
                   threshold_diff,
                   threshold_fit,
                   threshold_walen_slope,
                   threshold_flow_field_alignment,
                   n_trial_axes,
                   window_overlaps,
                   window_data,
                   window_frames,
                   window_starts,
                   window_vertical_directions,
                   contains_existing):
    window_mask = ~window_overlaps

    def execute_batched(function, dense=True):
        good_windows = torch.nonzero(window_mask).flatten()

        dataset = SlidingWindowDataset(window_data, good_windows, batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)

        batch_outputs = []

        for i_batch, batch_data in enumerate(dataloader):
            batch_data = batch_data[0]  # batch dimension is taken care of by dataset instead of dataloader
            batch_indices = good_windows[i_batch * batch_size:(i_batch + 1) * batch_size]

            if len(batch_indices) == 0:
                continue

            assert len(batch_indices) == len(batch_data), f"{len(batch_indices)} vs {len(batch_data)}"

            batch_output = function(batch_indices=batch_indices, batch_data=batch_data)
            if type(batch_output) != tuple:
                batch_output = (batch_output,)
            else:
                batch_mask = batch_output[0]
                batch_output = (batch_mask, *[x[batch_mask] for x in batch_output[1:]])
            batch_outputs.append(batch_output)

        combined_output = [torch.concat(x, dim=0) for x in zip(*batch_outputs)]
        new_mask = combined_output[0]
        assert len(new_mask) == len(good_windows)

        window_mask.scatter_(0, good_windows, new_mask)

        if len(combined_output) > 1:
            combined_output = combined_output[1:]
            new_good_windows = torch.nonzero(window_mask).flatten()

            if not dense:
                combined_output = tuple(
                    (torch.sparse_coo_tensor(new_good_windows.unsqueeze(0), x, (len(window_mask), *x.shape[1:]))
                     for x in combined_output))
            else:
                combined_output = tuple(
                    ((torch.zeros((len(window_mask),) + x.shape[1:], device=x.device, dtype=x.dtype) * torch.nan) \
                    .scatter(0, new_good_windows.reshape((-1, *([1] * (len(x.shape) - 1)))).expand(-1, *x.shape[1:]), x)
                     for x in combined_output))

            return combined_output[0] if len(combined_output) == 1 else combined_output

    execute_batched(partial(_get_inflection_point_mask,
                            window_vertical_directions=window_vertical_directions))

    # todo: fix error when execute_batched with return values is given 0 good indices
    if ~torch.any(window_mask):
        return None

    window_frame_quality = execute_batched(partial(_get_frame_quality_mask,
                                                   threshold_frame_quality=threshold_frame_quality,
                                                   window_frames=window_frames))

    if ~torch.any(window_mask):
        return None

    window_axes, window_error_diff, window_walen_slope, window_flow_field_alignment = execute_batched(
        partial(_find_axes,
                threshold_diff=threshold_diff,
                threshold_walen_slope=threshold_walen_slope,
                threshold_flow_field_alignment=threshold_flow_field_alignment,
                n_trial_axes=n_trial_axes,
                window_frames=window_frames,
                window_vertical_directions=window_vertical_directions))

    if ~torch.any(window_mask):
        return None

    window_map_Az, window_map_Bx, window_map_By, window_map_Bz, \
        window_map_Jx, window_map_Jy, window_map_Jz, \
        window_map_core_mask, window_error_fit = execute_batched(partial(_validate_map,
                                                                         window_axes=window_axes,
                                                                         window_frames=window_frames,
                                                                         window_walen_slope=window_walen_slope,
                                                                         nominal_window_length=nominal_window_length,
                                                                         sample_spacing=sample_spacing,
                                                                         threshold_walen_slope=threshold_walen_slope,
                                                                         threshold_fit=threshold_fit), dense=False)
    window_error_fit = window_error_fit.to_dense()

    good_indices = torch.nonzero(window_mask).flatten()
    for i in good_indices[torch.argsort(window_error_diff.index_select(0, good_indices), descending=False)]:
        start = window_starts[i]

        if torch.any(contains_existing[start:start + nominal_window_length]):
            window_mask[i] = False
            continue

        contains_existing[start:start + nominal_window_length] = True

    if ~torch.any(window_mask):
        return None

    temporal_scale = torch.ones_like(window_starts) * nominal_window_length * sample_spacing
    dx_dt = torch.linalg.norm(window_frames - (window_frames * window_axes).sum(dim=-1, keepdims=True) * window_axes,
                              dim=-1)
    spatial_scale = dx_dt * temporal_scale

    remaining_indices = torch.nonzero(window_mask).flatten()

    dataset = Dataset(
        data_vars=dict(
            window_start=(["event"], window_starts[remaining_indices].cpu().numpy()),
            window_length=(["event"], np.ones(len(remaining_indices), dtype=int) * nominal_window_length),
            temporal_scale=(["event"], temporal_scale[remaining_indices].cpu().numpy()),
            spatial_scale=(["event"], spatial_scale[remaining_indices].cpu().numpy()),
            residue_diff=(["event"], window_error_diff[remaining_indices].cpu().numpy()),
            residue_fit=(["event"], window_error_fit[remaining_indices].cpu().numpy()),
            walen_slope=(["event"], window_walen_slope[remaining_indices].cpu().numpy()),
            frame_quality=(["event"], window_frame_quality[remaining_indices].cpu().numpy()),
            flow_field_alignment=(["event"], window_flow_field_alignment[remaining_indices].cpu().numpy()),
            axis_x=(["event"], window_axes[remaining_indices, 0].cpu().numpy()),
            axis_y=(["event"], window_axes[remaining_indices, 1].cpu().numpy()),
            axis_z=(["event"], window_axes[remaining_indices, 2].cpu().numpy()),
            velocity_x=(["event"], window_frames[remaining_indices, 0].cpu().numpy()),
            velocity_y=(["event"], window_frames[remaining_indices, 1].cpu().numpy()),
            velocity_z=(["event"], window_frames[remaining_indices, 2].cpu().numpy()),
            map_Az=(["event", "y", "x"], window_map_Az.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_Bx=(["event", "y", "x"], window_map_Bx.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_By=(["event", "y", "x"], window_map_By.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_Bz=(["event", "y", "x"], window_map_Bz.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_Jx=(["event", "y", "x"], window_map_Jx.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_Jy=(["event", "y", "x"], window_map_Jy.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_Jz=(["event", "y", "x"], window_map_Jz.index_select(0, remaining_indices).to_dense().cpu().numpy()),
            map_core_mask=(
            ["event", "y", "x"], window_map_core_mask.index_select(0, remaining_indices).to_dense().cpu().numpy())
        ),
    )

    return dataset


def _pack_data(magnetic_field, velocity, density, gas_pressure):
    # all arrays must have the same number of samples
    n_sample = len(magnetic_field)
    assert n_sample == len(magnetic_field) == len(velocity) == len(density) == len(gas_pressure)

    # ensure all arrays have the correct shapes
    assert magnetic_field.shape == (n_sample, 3)
    assert velocity.shape == (n_sample, 3)
    assert density.shape == (n_sample,)
    assert gas_pressure.shape == (n_sample,)

    # convert all arrays to tensor so that numpy arrays can be given
    magnetic_field = torch.as_tensor(magnetic_field, dtype=torch.float32)
    velocity = torch.as_tensor(velocity, dtype=torch.float32)
    density = torch.as_tensor(density, dtype=torch.float32)
    gas_pressure = torch.as_tensor(gas_pressure, dtype=torch.float32).unsqueeze(1)

    # alfven velocity is calculated based on v_A = |B|/sqrt(n_p m_p mu_0)
    # constant factor 21.8114 assumes B is in nT and n_p is in cm^-3
    # and v_A should be in km/s
    alfven_velocity = (magnetic_field[:, :3] / torch.sqrt(density.unsqueeze(1))) * 21.8114
    alfven_velocity = torch.where(density[:, None] == 0, torch.inf, alfven_velocity)

    electric_field = -torch.cross(velocity, magnetic_field, dim=-1)

    # combine needed data into one tensor
    tensor = torch.concat([magnetic_field,
                           velocity,
                           alfven_velocity,
                           electric_field,
                           gas_pressure], dim=1)

    assert not torch.any(torch.isnan(
        tensor)), "Data contains missing values (nan). PyMFR does not automatically handle missing values. Instead, please keep track of which data points contain nans beforehand, interpolate or fill, and then remove flux ropes with intervals containing too many NaNs."

    return tensor


def _find_axes(batch_data,
               threshold_walen_slope,
               threshold_flow_field_alignment,
               batch_indices,
               threshold_diff,
               n_trial_axes,
               window_frames,
               window_vertical_directions):
    batch_vertical_directions = window_vertical_directions.index_select(0, batch_indices)
    batch_frames = window_frames.index_select(0, batch_indices)

    # extract individual physical quantities
    batch_magnetic_field = batch_data[:, :, :3]
    batch_gas_pressure = batch_data[:, :, 12]

    batch_normalized_potential = _calculate_normalized_potential(batch_magnetic_field, batch_vertical_directions)

    batch_axes, axis_residue = _minimize_axis_residue(batch_magnetic_field,
                                                      batch_vertical_directions,
                                                      batch_normalized_potential,
                                                      n_trial_axes)

    axial_component = (batch_magnetic_field @ batch_axes.unsqueeze(-1)).squeeze(-1)

    batch_velocity = batch_data[:, :, 3:6]
    batch_alfven_velocity = batch_data[:, :, 6:9]

    walen_slope, flow_field_alignment = _calculate_walen_slope(batch_frames, batch_alfven_velocity, batch_velocity)
    walen_slope_mask = torch.abs(walen_slope) <= threshold_walen_slope

    if threshold_flow_field_alignment is not None:
        walen_slope_mask = walen_slope_mask | ((torch.abs(flow_field_alignment) >= threshold_flow_field_alignment) & (
                    torch.abs(walen_slope) < 0.9))

    alpha = torch.where(torch.abs(walen_slope) < threshold_walen_slope, 0, walen_slope ** 2).unsqueeze(1)

    transverse_pressure = _calculate_transverse_pressure(torch.linalg.norm(batch_magnetic_field, dim=-1),
                                                         axial_component,
                                                         batch_gas_pressure,
                                                         alpha)

    axis_residue = _calculate_residue_diff(batch_normalized_potential, axial_component)
    error_diff = _calculate_residue_diff(batch_normalized_potential, transverse_pressure)

    # combine all the masks
    mask = walen_slope_mask & (axis_residue <= threshold_diff) & (error_diff <= threshold_diff)

    return mask, batch_axes, error_diff, walen_slope, flow_field_alignment


def _calculate_transverse_pressure(B_magnitude, B_transverse, Pgas, alpha):
    transverse_magnetic_pressure = (B_transverse * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
    isotropic_magnetic_pressure = (B_magnitude * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
    # calculate $p_t = B_z^2/2mu_0 + p/(1-alpha) + (alpha/(1-alpha)) B^2/2mu_0$
    # since alpha is constant, no need for extra factor of (1 - alpha) used in Chen et al
    # because \del^2 A' = \del^2 ((1 - alpha) A) = (1 - alpha) \del^2 A
    # and dPt'/dA' = (1/(1-alpha))dPt'/dA
    # also, \del^2 A = -\mu_0 j_z for any 2D field, field aligned flow or not,
    # so \del^2 A version of Pt is more useful than the one for \del^2 A'
    transverse_pressure = transverse_magnetic_pressure + (Pgas + alpha * isotropic_magnetic_pressure) / (1 - alpha)
    return transverse_pressure


def _get_inflection_point_mask(batch_data, batch_indices, window_vertical_directions):
    batch_vertical_directions = window_vertical_directions.index_select(0, batch_indices)
    batch_magnetic_field = batch_data[:, :, :3]

    batch_normalized_potential = _calculate_normalized_potential(batch_magnetic_field, batch_vertical_directions)

    return _count_inflection_points(batch_normalized_potential) == 1


def _get_frame_quality_mask(batch_data, batch_indices, threshold_frame_quality, window_frames):
    batch_frames = window_frames.index_select(0, batch_indices)
    batch_magnetic_field = batch_data[:, :, :3]
    batch_velocity = batch_data[:, :, 3:6]
    batch_electric_field = batch_data[:, :, 9:12]

    frame_quality = _calculate_frame_quality(batch_frames, batch_magnetic_field, batch_velocity, batch_electric_field)
    return frame_quality > threshold_frame_quality, frame_quality


def _validate_map(batch_indices,
                  batch_data,
                  window_axes,
                  window_frames,
                  window_walen_slope,
                  nominal_window_length,
                  sample_spacing,
                  threshold_walen_slope,
                  threshold_fit):
    batch_axes = window_axes.index_select(0, batch_indices)
    batch_frames = window_frames.index_select(0, batch_indices)
    batch_walen_slope = window_walen_slope.index_select(0, batch_indices)

    reconstructed_map = _generate_map(batch_data,
                                      batch_axes,
                                      batch_frames,
                                      batch_walen_slope,
                                      nominal_window_length,
                                      sample_spacing,
                                      threshold_walen_slope)

    mask_fit = reconstructed_map.error_fit <= threshold_fit

    y_observed = reconstructed_map.magnetic_potential.shape[-2] // 2

    height = torch.any(reconstructed_map.core_mask, dim=-1).to(float).sum(dim=-1)

    sign = torch.sign(reconstructed_map.magnetic_potential[:, y_observed, :].mean(dim=1))

    closed_values = torch.abs(
        torch.where(reconstructed_map.core_mask, reconstructed_map.magnetic_potential * sign[:, None, None], torch.nan))
    peak_closed_value = torch.nan_to_num(closed_values, -torch.inf).flatten(-2).amax(dim=-1)
    peak_observed_value = torch.nan_to_num(closed_values, -torch.inf)[:, y_observed, :].amax(dim=-1)
    minimum_closed_value = torch.nan_to_num(closed_values, torch.inf).flatten(-2).amin(dim=-1)
    minimum_observed_value = torch.nan_to_num(closed_values, torch.inf).flatten(-2).amin(dim=-1)

    height = torch.any(reconstructed_map.core_mask, dim=-1).to(float).sum(dim=-1)
    width = torch.any(reconstructed_map.core_mask, dim=-2).to(float).sum(dim=-1)

    mask_closed = (height > 3) & (width > 3) & (minimum_closed_value < peak_observed_value)

    mask_valid = ~torch.any(
        (torch.isnan(reconstructed_map.magnetic_potential) | torch.isinf(reconstructed_map.magnetic_potential)).flatten(
            -2), dim=-1)

    return mask_fit & mask_closed & mask_valid, \
        reconstructed_map.magnetic_potential, reconstructed_map.magnetic_field_x, reconstructed_map.magnetic_field_y, reconstructed_map.magnetic_field_z, \
        reconstructed_map.current_density_x, reconstructed_map.current_density_y, reconstructed_map.current_density_z, \
        reconstructed_map.core_mask, reconstructed_map.error_fit


def _generate_map(data,
                  axes,
                  frames,
                  walen_slope,
                  nominal_window_length,
                  sample_spacing,
                  threshold_walen_slope):
    batch_magnetic_field = data[:, :, :3]
    batch_gas_pressure = data[:, :, 12]

    alpha = walen_slope ** 2

    time_elapsed = nominal_window_length * sample_spacing
    dx_dt = torch.linalg.norm(frames - (frames * axes).sum(dim=-1, keepdims=True) * axes, dim=-1)

    dt = time_elapsed / data.shape[-2]
    dx = dx_dt * dt

    x_axis = F.normalize(-(frames - (frames * axes).sum(dim=-1, keepdim=True) * axes), dim=-1)
    y_axis = F.normalize(torch.cross(axes, x_axis, dim=-1), dim=-1)
    basis = torch.stack([x_axis, y_axis, axes], dim=-1)

    resolution = 15

    reconstructed_map = reconstruct_map(batch_magnetic_field @ basis, batch_gas_pressure, alpha=alpha,
                                        sample_spacing=dx, poly_order=2, resolution=resolution)

    return reconstructed_map


def _sliding_frames(tensor, window_length, window_step):
    magnetic_field = tensor[..., :3]
    electric_field = tensor[..., 9:12]

    Bx, By, Bz = magnetic_field[..., 0], magnetic_field[..., 1], magnetic_field[..., 2]
    coefficients = [[By ** 2 + Bz ** 2, -Bx * By, -Bx * Bz],
                    [-Bx * By, Bx ** 2 + Bz ** 2, -By * Bz],
                    [-Bx * Bz, -By * Bz, Bx ** 2 + By ** 2]]
    coefficient_matrix = torch.zeros(
        (*tensor.shape[:-2], int(math.floor((tensor.shape[-2] - window_length) / window_step + 1)), 3, 3),
        device=tensor.device, dtype=tensor.dtype)
    for i in range(3):
        for j in range(3):
            coefficient_matrix[..., i, j] = F.avg_pool1d(coefficients[i][j].unsqueeze(-2), window_length,
                                                         window_step).squeeze(-2)

    dependent_values = torch.cross(electric_field, magnetic_field, dim=-1)
    dependent_values = F.avg_pool1d(dependent_values.T, window_length, window_step).T

    try:
        frames = torch.linalg.lstsq(coefficient_matrix, dependent_values).solution
    except Exception:
        # lstsq appears to be buggy on GPU so do it on CPU if it fails
        frames = torch.linalg.lstsq(coefficient_matrix.cpu(), dependent_values.cpu()).solution.to(tensor.device)

    return frames


def _calculate_frame_quality(frames, magnetic_field, velocity, electric_field):
    electric_field = -torch.cross(velocity, magnetic_field, dim=2)

    remaining_electric_field = -torch.cross(velocity - frames.unsqueeze(1), magnetic_field, dim=2)

    frame_error = (remaining_electric_field ** 2).sum(dim=2).mean(dim=1) / (electric_field ** 2).sum(dim=2).mean(dim=1)
    frame_quality = torch.sqrt(1 - frame_error)

    return frame_quality


def _minimize_axis_residue(magnetic_field, vertical_directions, potential, n_trial_axes, return_residue=True):
    # use avg magnetic field as initial guess
    initial_guess = magnetic_field.mean(dim=-2)
    # ensure perpendicular to vertical direction
    initial_guess = initial_guess - (initial_guess * vertical_directions).sum(dim=-1,
                                                                              keepdims=True) * vertical_directions
    axes = F.normalize(initial_guess, dim=-1)

    axes, residue = _minimize_axis_residue_narrower(magnetic_field, vertical_directions, potential, n_trial_axes, axes)

    if return_residue:
        return axes, residue
    else:
        return axes


def _calculate_residue_map(magnetic_field, trial_axes, potential):
    difference_vectors = torch.stack(
        [_calculate_folding_differences(potential, magnetic_field[:, :, i]) for i in range(3)], dim=-1)

    differences = difference_vectors @ trial_axes.transpose(1, 2)
    axial = magnetic_field @ trial_axes.transpose(1, 2)

    min_axial, max_axial = torch.aminmax(axial, dim=1)
    axial_range = max_axial - min_axial
    error_diff = torch.sqrt(torch.mean((differences) ** 2, dim=1)) / axial_range

    return error_diff


def _minimize_axis_residue_narrower(magnetic_field, vertical_directions, potential, n_trial_axes, initial_axes):
    # generate trial axes by rotating <B> around the vertical axis (we expect z to be <B> +- 90 degrees around y, since <Bz> should be positive)
    alternative_directions = F.normalize(torch.cross(vertical_directions, initial_axes, dim=-1), dim=-1)
    angle = torch.linspace(-np.pi / 2, np.pi / 2, n_trial_axes, device=initial_axes.device).reshape(1, n_trial_axes, 1)
    trial_axes = F.normalize(
        initial_axes.unsqueeze(1) * torch.cos(angle) + alternative_directions.unsqueeze(1) * torch.sin(angle), dim=-1)

    # For determining the axis we only use magnetic field data, not gas pressure
    # This is because gas pressure error diff is not affected by rotations about the y axis
    # unlike Bz, which depends on the axis orientation
    # Bz itself is conserved along field lines, and this way we can compare the SIGN of Bz as well
    error_diff_axis = _calculate_residue_map(magnetic_field, trial_axes, potential)

    # select the best orientation
    residue, argmin = error_diff_axis.min(dim=-1)
    index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
    optimal_axes = trial_axes.gather(-2, index).squeeze(-2)
    return optimal_axes, residue


def _sliding_vertical_directions(scaled_tensor, window_length, window_step, windows, window_frames,
                                 window_avg_field_strength, batch_size, n_trial_axes):
    # use conv1d kernel to perform trapezoid integration
    integration_weight = torch.ones(window_length, dtype=scaled_tensor.dtype, device=scaled_tensor.device)
    integration_weight[0] = 1 / 2
    integration_weight[-1] = 1 / 2
    integration_weight = integration_weight / integration_weight.sum()
    integration_weight = integration_weight.reshape(1, 1, -1)
    window_avg_field = torch.conv1d(scaled_tensor[:, :3].T.unsqueeze(1), weight=integration_weight,
                                    stride=window_step).squeeze(1).T

    avg_field_direction = F.normalize(window_avg_field, dim=-1)
    path_direction = -F.normalize(window_frames, dim=-1)
    cross_product = torch.cross(avg_field_direction, path_direction, dim=-1)
    vertical_directions = F.normalize(cross_product, dim=-1)

    # compare ratio of average along guessed vertical direction and the one perpendicular. if less than 10x difference, not well determined
    perpendicular_direction = F.normalize(torch.cross(path_direction, vertical_directions, dim=-1), dim=-1)
    ratio = (window_avg_field * perpendicular_direction).sum(dim=-1) / window_avg_field_strength
    too_close = (ratio < 0.1) | (
                torch.norm(vertical_directions, dim=-1) == 0)  # also too close if norm of vertical direction is 0

    too_close_indices = torch.nonzero(too_close).flatten()

    for i in range(0, len(too_close_indices), batch_size):
        too_close_batch_indices = too_close_indices[i:i + batch_size]
        too_close_field = windows[too_close_batch_indices, :3].transpose(1, 2)
        too_close_mean_directions = avg_field_direction.index_select(0, too_close_batch_indices)
        too_close_frames = window_frames.index_select(0, too_close_batch_indices)
        too_close_path_directions = path_direction.index_select(0, too_close_batch_indices)

        # basis vectors will be the ones with the lowest component value in too_close_path_direction,
        # effectively the one with the lowest dot product and thus most perpendicular
        basis_vectors = torch.zeros_like(too_close_path_directions)
        basis_vectors.scatter_(dim=1, index=too_close_path_directions.abs().argmin(dim=-1, keepdim=True), value=1)

        # guaranteed perpendicular to too_close_path_directions and nonzero as long as velocity is nonzero
        arbitrary_perpendicular_directions = F.normalize(torch.cross(basis_vectors, too_close_path_directions, dim=-1),
                                                         dim=-1)

        # to be used for linear combinations for vertical trials
        alternative_directions = F.normalize(
            torch.cross(arbitrary_perpendicular_directions, too_close_path_directions, dim=-1), dim=-1)

        n_vertical_trials = 180

        angle = torch.linspace(-np.pi / 2, np.pi / 2, n_vertical_trials,
                               device=arbitrary_perpendicular_directions.device).reshape(1, n_vertical_trials, 1)
        possible_vertical_directions = F.normalize(
            arbitrary_perpendicular_directions.unsqueeze(1) * torch.cos(angle) + alternative_directions.unsqueeze(
                1) * torch.sin(angle), dim=-1)

        # ensure that x unit points opposite direction of frames
        x_unit = F.normalize(torch.cross(possible_vertical_directions, too_close_mean_directions.unsqueeze(1), dim=-1),
                             dim=-1)
        x_unit_sign = -torch.sign((x_unit * too_close_frames.unsqueeze(1)).sum(dim=-1, keepdim=True))
        possible_vertical_directions *= x_unit_sign

        n_too_close = len(too_close_batch_indices)

        # number processed = n in vertical axis batch * n too close
        # -> n in axis batch = number processed / n too close
        # max number processed (ideal for batch processing) would be
        # total number of trial vertical axes * n too close
        # but it cant be more than batch size
        # also must have minimum axis batch size 1 to get anywhere
        # hence the following line of code makes sense
        axis_batch_step = max(1, min(n_vertical_trials * n_too_close, batch_size) // n_too_close)

        minimum_residues = []  # best residue attainable for each vertical axis

        for i_axis_batch in range(0, n_vertical_trials, axis_batch_step):
            axis_batch_size = min(i_axis_batch + axis_batch_step, n_vertical_trials) - i_axis_batch

            # looks like [B1, B1, B1, B2, B2, B2] if axis_batch_size = 3 and n_too_close=2
            axis_batch_field = torch.repeat_interleave(too_close_field, axis_batch_size, dim=0)

            # looks like [axis1a, axis1b, axis1c, axis2a, axis2b, axis2c]
            axis_batch_directions = possible_vertical_directions[:, i_axis_batch:i_axis_batch + axis_batch_size,
                                    :].reshape(-1, 3)

            axis_batch_potential = _calculate_normalized_potential(axis_batch_field,
                                                                   axis_batch_directions)
            axis_batch_inflection_mask = (_count_inflection_points(axis_batch_potential) == 1) & (
                        torch.abs(axis_batch_potential[..., -1]) < torch.abs(axis_batch_potential).amax(dim=-1) * .1)

            axis_batch_min_residue = torch.ones(len(axis_batch_directions),
                                                dtype=axis_batch_directions.dtype,
                                                device=axis_batch_directions.device) * torch.inf

            if torch.any(axis_batch_inflection_mask):
                axis_batch_min_residue[axis_batch_inflection_mask] = \
                _minimize_axis_residue(axis_batch_field[axis_batch_inflection_mask],
                                       axis_batch_directions[axis_batch_inflection_mask],
                                       axis_batch_potential[axis_batch_inflection_mask],
                                       n_trial_axes)[1]

            # each entry of minimum_residues looks like
            # [[residue1a, residue1b, residue1c], [residue2a, residue2b, residue2c]]
            minimum_residues.append(axis_batch_min_residue.reshape(n_too_close, axis_batch_size))

        # looks like [[residue1a, ..., residue1z], [residue2a, ..., residue2z]]
        min_residue = torch.concat(minimum_residues, dim=1)
        assert min_residue.shape == (n_too_close, n_vertical_trials)
        argmin = min_residue.argmin(dim=-1)
        index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
        vertical_directions.scatter_(0, too_close_batch_indices.unsqueeze(1).expand(-1, 3),
                                     possible_vertical_directions.gather(-2, index).squeeze(-2))
    return vertical_directions


def _get_sliding_windows(window_lengths, window_steps):
    assert list(sorted(window_lengths)) == list(sorted(set(window_lengths))), "window lengths must be unique"
    assert len(window_lengths) == len(window_steps), "must have same number of window steps and lengths"

    # use sliding windows sorted by longest window length to shortest
    # because the algorithm gives preference to longer MFRs candidates over shorter ones
    # since long MFRs tend to show up as a bunch of small MFRs (especially ICMEs)
    return list(reversed(sorted(zip(window_lengths, window_steps), key=lambda x: x[0])))


def _calculate_walen_slope(batch_frames, alfven_velocity, velocity):
    remaining_flow = (velocity - batch_frames.unsqueeze(1))

    # simple linear regression slope with y intercept fixed at 0
    walen_slope = (remaining_flow.flatten(1) * alfven_velocity.flatten(1)).sum(dim=1) / (
                alfven_velocity.flatten(1) ** 2).sum(dim=1)

    # this is equivalent to pearson correlation except forcing y intercept to be 0
    field_alignment = (
                F.normalize(remaining_flow.flatten(1), dim=1) * F.normalize(alfven_velocity.flatten(1), dim=1)).sum(
        dim=1)

    return walen_slope, field_alignment


def _count_inflection_points(potential):
    # stricter smoothing than the original algorithm,
    # because this implementation relies more heavily on having only a single inflection point
    # since it does no trimming of the window size.
    # based on my testing, doing this does not drastically reduce the number of quality flux ropes
    # but it does greatly increase the quality of the flux ropes that are detected - H. Farooki

    kernel_size = max(potential.shape[-1] // 10, 1)

    if kernel_size % 2 == 0:
        kernel_size += 1

    # simple moving average
    smoothed = F.avg_pool1d(potential,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            count_include_pad=False,
                            stride=1)

    is_sign_change = (torch.diff(torch.sign(torch.diff(smoothed))) != 0).to(int)

    return torch.sum(is_sign_change, dim=-1)


def _calculate_normalized_potential(magnetic_field, vertical_directions):
    # calculate the A array/magnetic flux function/potential using $A(x) = -\int_{x_0}^x B_y dx$
    batch_magnetic_field_y = (magnetic_field @ vertical_directions.unsqueeze(-1)).squeeze(-1)
    unscaled_potential = torch.zeros(batch_magnetic_field_y.shape, device=batch_magnetic_field_y.device)
    unscaled_potential[..., 1:] = torch.cumulative_trapezoid(batch_magnetic_field_y)

    # use the potential peaks as the inflection points
    inflection_points = unscaled_potential[..., 1:-1].abs().argmax(dim=-1) + 1

    peak_values = unscaled_potential.gather(-1, inflection_points.unsqueeze(-1))
    normalized_potential = unscaled_potential / peak_values

    return normalized_potential
