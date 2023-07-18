import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm as tqdm
import math
from functools import partial

from pymfr.residue import _calculate_residue_diff, _calculate_residue_fit, _calculate_folding_differences


class SlidingWindowDataset(Dataset):
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




def detect_flux_ropes(magnetic_field,
                      velocity,
                      density,
                      gas_pressure,
                      batch_size,
                      window_lengths,
                      window_steps,
                      min_strength,
                      frame_type="vht",
                      threshold_frame_quality=0.9,
                      threshold_diff=0.12,
                      threshold_fit=0.14,
                      threshold_walen=0.3,
                      threshold_flow_field_alignment=None,
                      n_axis_iterations=1,
                      n_trial_axes=180 // 4,
                      max_processing_resolution=64,
                      min_positive_axial_component=0.95,
                      cuda=True,
                      axis_optimizer=None,
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
    :param min_strength: Minimum <B> for sliding window candidates in nT.
    :param frame_type: The method used to estimate the propagation direction of the flux rope.
    Options:
    - "mean_velocity"
        Simply takes the average velocity over the window. Fastest but least precise.
    - "vht":
        Finds the frame that minimizes the averaged electric field magnitude squared.
    :param threshold_frame_quality: Minimum correlation coefficient for the HT frame (Khabrov and Sonnerup, 1998). Default 0.9.
    :param threshold_diff: The maximum allowable R_diff
    :param threshold_fit: The maximum allowable R_fit
    :param threshold_walen: The Walen slope threshold to determine if there is significant Alfvenicity.
    :param threshold_flow_field_alignment: The threshold for the average absolute value of the
    dot product between normalized remaining flow and normalized magnetic field if threshold_walen is not met.
    If flux ropes with field aligned flows are not desired, then set to None.
    Default None.
    :param n_axis_iterations: Number of iterations for axis determination algorithm.
    Each iteration `i in {0, ..., n_axis_iterations - 1}` will scan a region of angular width $\pi/2^i$
    centered around the previous best orientation (using avg magnetic field direction as an initial guess)
    with n_trial_axes evenly spaced guess orientations.
    This feature is different from the original GS automated detection algorithm. Similar behavior can be obtained with `n_axis_iterations = 1`.
    :param n_trial_axes: The number of evenly spaced guess orientations used for the axis determination algorithm.
    Odd number is desirable so that one of the trial axes includes the previous iteration's best orientation (starting with avg field direction).
    :param max_processing_resolution: Scale sliding window data down to approximately this size (if larger) before generating sliding windows to reduce memory usage.
    :param min_positive_axial_component: Percentage of Bz to require to be positive. Default 0.95 (i.e. 95%).
    For behavior of original GS algorithm, set to 0.
    :param cuda: Whether to use the GPU
    :param axis_optimizer: Function of (magnetic_field, frames, vertical_directions) that returns optimized axes. Intended for deep learning based axis optimization.
    :return: A DataFrame describing the detected flux ropes.
    """

    full_tensor = _pack_data(magnetic_field, velocity, density, gas_pressure)

    if cuda:
        full_tensor = full_tensor.cuda()

    # these are updated as the algorithm runs
    # contains_existing keeps track of samples that were already confirmed as MFRs
    # for longer durations so that shorter durations do not attempt to overwrite them
    contains_existing = torch.zeros(len(magnetic_field), dtype=torch.bool, device=full_tensor.device)
    remaining_events = list()

    sliding_windows = _get_sliding_windows(window_lengths, window_steps)

    for nominal_window_length, window_step in (sliding_windows if not progress_bar else tqdm.tqdm(sliding_windows)):
        scaled_tensor = full_tensor
        downsample_factor = 1 if max_processing_resolution is None else max(nominal_window_length // max_processing_resolution, 1) 
        window_length = nominal_window_length // downsample_factor
        if downsample_factor > 1:
            scaled_tensor = F.avg_pool1d(full_tensor.T, downsample_factor, downsample_factor).T
            window_step = max(1, window_step // downsample_factor)

        window_avg_field_strength = F.avg_pool1d(torch.norm(scaled_tensor[:, :3], dim=1).unsqueeze(0), window_length, window_step).squeeze(0)
        window_overlaps = F.max_pool1d(F.max_pool1d(contains_existing.to(float).unsqueeze(0), downsample_factor, downsample_factor), window_length, window_step).squeeze(0) > 0
            
        # generate the sliding windows as a tensor view, which is only actually loaded into memory when copied on demand
        window_data = scaled_tensor.unfold(size=window_length, step=window_step, dimension=0)
        
        window_frames = _sliding_frames(scaled_tensor, window_length, window_step, frame_type)
        
        window_avg_field = _sliding_average_field(scaled_tensor, window_length, window_step)
        
        window_vertical_directions = _sliding_vertical_directions(window_data,
                                                                  window_frames,
                                                                  window_avg_field,
                                                                  batch_size,
                                                                  n_axis_iterations,
                                                                  n_trial_axes,
                                                                  axis_optimizer)

        window_starts = torch.arange(len(window_data), device=scaled_tensor.device) * window_step * downsample_factor

        window_mask = (window_avg_field_strength >= min_strength) & ~window_overlaps
        
        batch_outputs = []

        batch_functions = [
            partial(_get_inflection_point_mask,
                    window_vertical_directions=window_vertical_directions),
            partial(_get_frame_quality_mask,
                    threshold_frame_quality=threshold_frame_quality,
                    window_frames=window_frames),
            partial(_get_alfvenicity_mask,
                    threshold_walen=threshold_walen,
                    threshold_flow_field_alignment=threshold_flow_field_alignment,
                    window_frames=window_frames),
            partial(_find_axes,
                    threshold_diff=threshold_diff,
                    n_axis_iterations=n_axis_iterations,
                    n_trial_axes=n_trial_axes,
                    min_positive_axial_component=min_positive_axial_component,
                    axis_optimizer=axis_optimizer,
                    window_vertical_directions=window_vertical_directions,
                    batch_outputs=batch_outputs)
        ]

        for function in batch_functions:

            good_windows = torch.nonzero(window_mask).flatten()
            
            masks = []

            dataset = SlidingWindowDataset(window_data, good_windows, batch_size)
            dataloader = DataLoader(dataset, num_workers=0)

            for i_batch, batch_data in enumerate(dataloader):
                batch_data = batch_data[0]  # batch dimension is taken care of by dataset instead of dataloader
                batch_indices = good_windows[i_batch * batch_size:(i_batch + 1) * batch_size]

                if len(batch_indices) == 0:
                    continue

                assert len(batch_indices) == len(batch_data), f"{len(batch_indices)} vs {len(batch_data)}"
            
                masks.append(function(batch_indices=batch_indices, batch_data=batch_data))
            assert sum(len(x) for x in masks) == len(good_windows)
            
            new_mask = torch.concat(masks, dim=0)
            window_mask.put_(good_windows, new_mask & window_mask.index_select(0, good_windows))

        # candidate cleanup done at the end of processing all batches with a given window length
        
        event_candidates = _process_candidates(threshold_fit,
                           nominal_window_length,
                           window_data,
                           window_frames,
                           window_vertical_directions,
                           window_starts,
                           batch_outputs)

        remaining_events.extend(_cleanup_candidates(contains_existing, event_candidates).cpu().numpy())

    remaining_events = list(sorted(remaining_events, key=lambda x: x[1]))
    df = pd.DataFrame(remaining_events, columns=["start",
                                                   "end",
                                                   "duration",
                                                   "error_diff", 
                                                   "error_fit",
                                                   "walen_slope",
                                                   "frame_quality",
                                                   "flow_field_alignment",
                                                   "axis_x",
                                                   "axis_y",
                                                   "axis_z",
                                                   "frame_x",
                                                   "frame_y",
                                                   "frame_z"])
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["duration"] = df["duration"].astype(int)
    return df

def _find_axes(batch_data,
                batch_indices,
                threshold_diff,
                n_axis_iterations,
                n_trial_axes,
                min_positive_axial_component,
                axis_optimizer,
                window_vertical_directions,
                batch_outputs):
    batch_vertical_directions = window_vertical_directions.index_select(0, batch_indices)

    # extract individual physical quantities
    batch_magnetic_field = batch_data[:, :, :3]
    batch_gas_pressure = batch_data[:, :, 12]
            

    batch_normalized_potential, inflection_points = _calculate_normalized_potential(batch_magnetic_field, batch_vertical_directions)
            
    batch_axes, axis_residue = _minimize_axis_residue(batch_magnetic_field,
                                                batch_vertical_directions,
                                                batch_normalized_potential,
                                                n_axis_iterations,
                                                n_trial_axes,
                                                axis_optimizer)

    axial_component = (batch_magnetic_field @ batch_axes.unsqueeze(-1)).squeeze(-1)
            
    # calculate $p_t = B_z^2/2mu_0 + p$
    transverse_magnetic_pressure = (axial_component * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
    transverse_pressure = batch_gas_pressure + transverse_magnetic_pressure
            
    # require the inflection point pressure to be within the top 85%
    peak_pressure = transverse_pressure[torch.arange(len(transverse_pressure), device=batch_data.device), inflection_points]
    pressure_peak_mask = peak_pressure > torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")

    # require a certain percentage of Bz to be positive
    mask_positive_Bz = (axial_component > 0).to(torch.float32).mean(dim=-1) > min_positive_axial_component

    error_diff = _calculate_residue_diff(batch_normalized_potential, transverse_pressure)

    # combine all the masks
    mask = pressure_peak_mask & mask_positive_Bz & (axis_residue <= threshold_diff) & (error_diff <= threshold_diff)

    batch_outputs.append((mask, batch_indices, batch_axes, error_diff))

    return mask

def _get_alfvenicity_mask(batch_data, batch_indices, threshold_walen, threshold_flow_field_alignment, window_frames):
    batch_frames = window_frames.index_select(0, batch_indices)
    batch_velocity = batch_data[:, :, 3:6]
    batch_alfven_velocity = batch_data[:, :, 6:9]

    walen_slope, flow_field_alignment = _calculate_alfvenicity(batch_frames, batch_alfven_velocity, batch_velocity)
    alfvenicity_mask = torch.abs(walen_slope) <= threshold_walen

    if threshold_flow_field_alignment is not None:
        alfvenicity_mask = alfvenicity_mask | (torch.abs(flow_field_alignment) >= threshold_flow_field_alignment)

    return alfvenicity_mask
    

def _get_inflection_point_mask(batch_data, batch_indices, window_vertical_directions):
    batch_vertical_directions = window_vertical_directions.index_select(0, batch_indices)
    batch_magnetic_field = batch_data[:, :, :3]
            
    batch_normalized_potential, inflection_points = _calculate_normalized_potential(batch_magnetic_field, batch_vertical_directions)

    return _count_inflection_points(batch_normalized_potential) == 1


def _get_frame_quality_mask(batch_data, batch_indices, threshold_frame_quality, window_frames):
    batch_frames = window_frames.index_select(0, batch_indices)
    batch_magnetic_field = batch_data[:, :, :3]
    batch_velocity = batch_data[:, :, 3:6]
    batch_electric_field = batch_data[:, :, 9:12]

    frame_quality = _calculate_frame_quality(batch_frames, batch_magnetic_field, batch_velocity, batch_electric_field)
    return frame_quality > threshold_frame_quality


def _process_candidates(threshold_fit,
                       nominal_window_length,
                       window_data,
                       window_frames,
                       window_vertical_directions,
                       window_starts,
                       batch_outputs):
    event_candidates = list()
    

    for mask, batch_indices, batch_axes, batch_error_diff in batch_outputs:
        nonzero = torch.nonzero(mask).flatten()

        if len(nonzero) == 0:
            continue

        batch_axes = batch_axes[nonzero]
        batch_error_diff = batch_error_diff[nonzero]

        nonzero_indices = batch_indices[nonzero]
        batch_data = window_data[nonzero_indices].transpose(1, 2)
        batch_frames = window_frames[nonzero_indices]
        batch_starts = window_starts[nonzero_indices]

        batch_magnetic_field = batch_data[:, :, :3]
        batch_velocity = batch_data[:, :, 3:6]
        batch_alfven_velocity = batch_data[:, :, 6:9]
        batch_electric_field = batch_data[:, :, 9:12]
        batch_gas_pressure = batch_data[:, :, 12]
        axial_component = (batch_magnetic_field @ batch_axes.unsqueeze(-1)).squeeze(-1)
        transverse_magnetic_pressure = (axial_component * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
        transverse_pressure = batch_gas_pressure + transverse_magnetic_pressure

        batch_potential, _ = _calculate_normalized_potential(batch_magnetic_field,
                                                                                   window_vertical_directions[nonzero_indices])
        
        batch_error_fit = _calculate_residue_fit(batch_potential, transverse_pressure)
        batch_frame_quality = _calculate_frame_quality(batch_frames, batch_magnetic_field, batch_velocity, batch_electric_field)
        batch_walen_slope, batch_flow_field_alignment = _calculate_alfvenicity(batch_frames, batch_alfven_velocity, batch_velocity)
        
        
        batch_candidates = torch.stack([batch_starts,
                         batch_starts + nominal_window_length - 1,
                         torch.ones_like(batch_starts) * nominal_window_length,
                         batch_error_diff,
                         batch_error_fit,
                         batch_walen_slope,
                         batch_frame_quality,
                         batch_flow_field_alignment,
                         batch_axes[:, 0], batch_axes[:, 1], batch_axes[:, 2],
                         batch_frames[:, 0], batch_frames[:, 1], batch_frames[:, 2]], dim=-1)
        assert batch_candidates.shape == (len(batch_starts), 14)
        if threshold_fit is not None:
            batch_candidates = batch_candidates[batch_error_fit <= threshold_fit]
        event_candidates.append(batch_candidates)

    if len(event_candidates) == 0:
        return torch.empty(size=(0, 14))
    return torch.concat(event_candidates, dim=0)


def _sliding_frames(tensor, duration, window_step, frame_type):
    def avg(x):
        return F.avg_pool1d(x.T, duration, window_step).T


    velocity = tensor[..., 3:6]
    if frame_type == "mean_velocity":
        return avg(velocity)    

    assert frame_type == "vht"

    magnetic_field = tensor[..., :3]
    electric_field = tensor[..., 9:12]

    Bx, By, Bz = magnetic_field[..., 0], magnetic_field[..., 1], magnetic_field[..., 2]
    coefficients = [[By ** 2 + Bz ** 2, -Bx * By, -Bx * Bz],
        [-Bx * By, Bx ** 2 + Bz ** 2, -By * Bz],
        [-Bx * Bz, -By * Bz, Bx ** 2 + By ** 2]]
    coefficient_matrix = torch.zeros((*tensor.shape[:-2], int(math.floor((tensor.shape[-2] - duration) / window_step + 1)), 3, 3), device=tensor.device, dtype=tensor.dtype)
    for i in range(3):
        for j in range(3):
            coefficient_matrix[..., i, j] = F.avg_pool1d(coefficients[i][j].unsqueeze(-2), duration, window_step).squeeze(-2)
    
    dependent_values = torch.cross(electric_field, magnetic_field, dim=-1)
    dependent_values = F.avg_pool1d(dependent_values.T, duration, window_step).T

    try:
        frames = torch.linalg.solve(coefficient_matrix, dependent_values)
    except Exception:
        # sometimes it is not well determined, but we want some solution anyway (if it is garbage the event will be discarded by the quality threshold)
        # lstsq appears to be buggy on GPU so do it on CPU
        frames = torch.linalg.lstsq(coefficient_matrix.cpu(), dependent_values.cpu()).solution.to(tensor.device)
    
    return frames


def _sliding_average_field(tensor, duration, window_step):
    # use conv1d kernel to perform trapezoid integration
    integration_weight = torch.ones(duration, dtype=tensor.dtype, device=tensor.device)
    integration_weight[0] = 1/2
    integration_weight[-1] = 1/2
    integration_weight = integration_weight / integration_weight.sum()
    integration_weight = integration_weight.reshape(1, 1, -1)
    return torch.conv1d(tensor[:, :3].T.unsqueeze(1), weight=integration_weight, stride=window_step).squeeze(1).T


def _calculate_frame_quality(frames, magnetic_field, velocity, electric_field):
    electric_field = -torch.cross(velocity, magnetic_field, dim=2)
    
    remaining_electric_field = -torch.cross(velocity - frames.unsqueeze(1), magnetic_field, dim=2)

    frame_error = (remaining_electric_field ** 2).sum(dim=2).mean(dim=1) / (electric_field ** 2).sum(dim=2).mean(dim=1)
    frame_quality = torch.sqrt(1 - frame_error)

    return frame_quality


def _minimize_axis_residue(magnetic_field, vertical_directions, potential, n_axis_iterations, n_trial_axes, axis_optimizer, return_residue=True):
    if axis_optimizer is not None:
        axes = axis_optimizer(magnetic_field, vertical_directions)
        if return_residue:
            residue = _calculate_residue_map(magnetic_field, axes.unsqueeze(1), potential).squeeze(-1)
    else:
        # use avg magnetic field as initial guess
        initial_guess = magnetic_field.mean(dim=-2)
        # ensure perpendicular to vertical direction
        initial_guess = initial_guess - (initial_guess * vertical_directions).sum(dim=-1, keepdims=True) * vertical_directions
        axes = F.normalize(initial_guess, dim=-1)
        
        for i in range(n_axis_iterations):
            spread = 1 / (2 ** i)
            axes, residue = _minimize_axis_residue_narrower(magnetic_field, vertical_directions, potential, n_trial_axes, axes, spread)

    if return_residue:
        return axes, residue
    else:
        return axes
    

def _calculate_residue_map(magnetic_field, trial_axes, potential):
    difference_vectors = torch.stack([_calculate_folding_differences(potential, magnetic_field[:, :, i]) for i in range(3)], dim=-1)
    
    differences = difference_vectors @ trial_axes.transpose(1, 2)
    axial = magnetic_field @ trial_axes.transpose(1, 2)

    min_axial, max_axial = torch.aminmax(axial, dim=1)
    axial_range = max_axial - min_axial
    error_diff = torch.sqrt(torch.mean(differences ** 2, dim=1) / 2) / axial_range

    return error_diff


def _minimize_axis_residue_narrower(magnetic_field, vertical_directions, potential, n_trial_axes, initial_axes, spread):
    # generate trial axes by rotating <B> around the vertical axis (we expect z to be <B> +- 90 degrees around y, since <Bz> should be positive)
    alternative_directions = F.normalize(torch.cross(vertical_directions, initial_axes, dim=-1), dim=-1)
    angle = torch.linspace(-np.pi / 2 * spread, np.pi / 2 * spread, n_trial_axes, device=initial_axes.device).reshape(1, n_trial_axes, 1)
    trial_axes = F.normalize(initial_axes.unsqueeze(1) * torch.cos(angle) + alternative_directions.unsqueeze(1) * torch.sin(angle), dim=-1)

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


def _sliding_vertical_directions(windows, window_frames, window_avg_field, batch_size, n_axis_iterations, n_trial_axes, axis_optimizer):
    avg_field_direction = F.normalize(window_avg_field, dim=-1)
    path_direction = -F.normalize(window_frames, dim=-1)
    cross_product = torch.cross(avg_field_direction, path_direction, dim=-1)
    vertical_directions = F.normalize(cross_product, dim=-1)
    
    # compare ratio of average along guessed vertical direction and the one perpendicular. if less than 10x difference, not well determined 
    perpendicular_direction = F.normalize(torch.cross(path_direction, vertical_directions, dim=-1), dim=-1)
    ratio = ((window_avg_field * vertical_directions).sum(dim=-1) / (window_avg_field * perpendicular_direction).sum(dim=-1)).abs()
    too_close = (ratio > 0.1) | (torch.norm(vertical_directions, dim=-1) == 0)  # also too close if norm of vertical direction is 0

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
        basis_vectors = basis_vectors.scatter(dim=1, index=too_close_path_directions.abs().argmin(dim=-1, keepdim=True), value=1)

        # guaranteed perpendicular to too_close_path_directions and nonzero as long as velocity is nonzero
        arbitrary_perpendicular_directions = F.normalize(torch.cross(basis_vectors, too_close_path_directions, dim=-1), dim=-1)
        
        # to be used for linear combinations for vertical trials
        alternative_directions = F.normalize(torch.cross(arbitrary_perpendicular_directions, too_close_path_directions, dim=-1), dim=-1)
        
        n_vertical_trials = 180

        angle = torch.linspace(-np.pi / 2, np.pi / 2, n_vertical_trials, device=arbitrary_perpendicular_directions.device).reshape(1, n_vertical_trials, 1)
        possible_vertical_directions = F.normalize(arbitrary_perpendicular_directions.unsqueeze(1) * torch.cos(angle) + alternative_directions.unsqueeze(1) * torch.sin(angle), dim=-1)
        
        # ensure that x unit points opposite direction of frames
        x_unit = F.normalize(torch.cross(possible_vertical_directions, too_close_mean_directions.unsqueeze(1), dim=-1), dim=-1)
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
            axis_batch_directions = possible_vertical_directions[:, i_axis_batch:i_axis_batch + axis_batch_size, :].reshape(-1, 3)
            
            axis_batch_potential, _ = _calculate_normalized_potential(axis_batch_field,
                                                                   axis_batch_directions)
            axis_batch_inflection_mask = _count_inflection_points(axis_batch_potential) == 1

            axis_batch_min_residue = torch.ones(len(axis_batch_directions),
                                                dtype=axis_batch_directions.dtype,
                                                device=axis_batch_directions.device) * torch.inf
            
            if torch.any(axis_batch_inflection_mask):
                axis_batch_min_residue[axis_batch_inflection_mask] = _minimize_axis_residue(axis_batch_field[axis_batch_inflection_mask],
                                                        axis_batch_directions[axis_batch_inflection_mask],
                                                        axis_batch_potential[axis_batch_inflection_mask],
                                                        n_axis_iterations,
                                                        n_trial_axes,
                                                        axis_optimizer)[1]
    

            # each entry of minimum_residues looks like
            # [[residue1a, residue1b, residue1c], [residue2a, residue2b, residue2c]]
            minimum_residues.append(axis_batch_min_residue.reshape(n_too_close, axis_batch_size))
        
        # looks like [[residue1a, ..., residue1z], [residue2a, ..., residue2z]]
        min_residue = torch.concat(minimum_residues, dim=1)
        assert min_residue.shape == (n_too_close, n_vertical_trials)
        argmin = min_residue.argmin(dim=-1)
        index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
        vertical_directions.put_(too_close_batch_indices.unsqueeze(1).expand(-1, 3), possible_vertical_directions.gather(-2, index).squeeze(-2))
    return vertical_directions


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
    tensor = torch.concat([magnetic_field, velocity, alfven_velocity, electric_field, gas_pressure], dim=1)

    assert not torch.any(torch.isnan(tensor)), "Data contains missing values (nan). PyMFR does not automatically handle missing values. Instead, please keep track of which data points contain nans beforehand, interpolate or fill, and then remove flux ropes with intervals containing too many NaNs."

    return tensor


def _get_sliding_windows(window_lengths, window_steps):
    assert list(sorted(window_lengths)) == list(sorted(set(window_lengths))), "window lengths must be unique"
    assert len(window_lengths) == len(window_steps), "must have same number of window steps and lengths"

    # use sliding windows sorted by longest duration to shortest
    # because the algorithm gives preference to longer MFRs candidates over shorter ones
    # since long MFRs tend to show up as a bunch of small MFRs (especially ICMEs)
    return list(reversed(sorted(zip(window_lengths, window_steps), key=lambda x: x[0])))


 
@torch.jit.script
def _cleanup_candidates(contains_existing, event_candidates):
    # sort by difference residue (instead of end time, as the original algorithm did)
    event_candidates = event_candidates[torch.argsort(event_candidates[:, 3], descending=True)]

    mask = torch.ones(len(event_candidates), device=event_candidates.device, dtype=torch.bool)
    
    for i in range(len(event_candidates)):
        event = event_candidates[i]

        start = int(event[0])
        duration = int(event[2])

        if torch.any(contains_existing[start:start + duration]):
            mask[i] = False
            continue

        contains_existing[start:start + duration] = True

    return event_candidates[mask]    



def _sliding_alfvenicity(window_length, window_step, alfven_velocity, velocity, frames):
    def avg(x):
        return F.avg_pool1d(x.T, window_length, window_step).T
    
    def dot(x, y):
        return (x * y).sum(dim=-1)
    
    def norm(x):
        return torch.norm(x, dim=-1)

    # <M_A^2> = <(v - v_HT)^2/v_A^2> = <v^2/v_A^2> - 2v_HT <v/v_A^2> + v_HT^2 <1/v_A^2>
    alfven_speed = norm(alfven_velocity, dim=-1)
    alfvenicity = avg(norm(velocity) ** 2 / alfven_speed ** 2) \
            - 2 * dot(frames, avg(velocity / alfven_speed ** 2)) \
            + norm(frames, dim=-1) ** 2 * avg(alfven_speed ** -2)

    # <(v - v_HT) dot B> = <v dot B> - <v_HT dot B>
    # <v_HT dot B> = <v_HT,x Bx + v_HT,y By + v_HT,z Bz> = v_HT dot <B>


    field_alignment = (F.normalize(remaining_flow, dim=2) * F.normalize(alfven_velocity, dim=2)).sum(dim=2).mean(dim=1)
    return walen_slope, field_alignment



def _calculate_alfvenicity(batch_frames, alfven_velocity, velocity):
    remaining_flow = (velocity - batch_frames.unsqueeze(1)).squeeze(1)
    d_flow = remaining_flow - remaining_flow.mean(dim=1, keepdim=True)
    d_alfven = alfven_velocity - alfven_velocity.mean(dim=1, keepdim=True)
    walen_slope = (d_flow.flatten(1) * d_alfven.flatten(1)).sum(dim=1) / (d_alfven.flatten(1) ** 2).sum(dim=1)
    field_alignment = (F.normalize(remaining_flow, dim=2) * F.normalize(alfven_velocity, dim=2)).sum(dim=2).mean(dim=1)
    return walen_slope, field_alignment


def _count_inflection_points(potential):
    # stricter smoothing than the original algorithm,
    # because this implementation relies more heavily on having only a single inflection point
    # since it does no trimming of the window size.
    # based on my testing, doing this does not drastically reduce the number of quality flux ropes
    # but it does greatly increase the quality of the flux ropes that are detected - H. Farooki

    smoothing_scale = max(potential.shape[-1] // 10, 1)

    # simple moving average
    smoothed = F.avg_pool1d(potential, kernel_size=smoothing_scale, padding=smoothing_scale // 2, count_include_pad=False, stride=1)

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
    
    return normalized_potential, inflection_points

