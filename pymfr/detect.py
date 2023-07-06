from typing import Iterator
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm as tqdm
from torch.utils.data import Dataset, DataLoader
import math

from scipy.signal import savgol_coeffs

from pymfr.axis import _rotate_field, calculate_residue_map
from pymfr.frame import estimate_ht_frame

from pymfr.residue import _calculate_residue_diff, _calculate_residue_fit


def detect_flux_ropes(magnetic_field,
                      velocity,
                      density,
                      gas_pressure,
                      batch_size,
                      window_lengths,
                      window_steps,
                      min_strength,
                      frame_type="vht",
                      threshold_diff=0.12,
                      threshold_fit=0.14,
                      threshold_walen=0.3,
                      n_axis_iterations=1,
                      n_trial_axes=180 // 4,
                      max_axis_determination_resolution=None,
                      min_positive_axial_component=0.95,
                      cuda=True,
                      axis_optimizer=None):
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
    :param threshold_diff: The maximum allowable R_diff
    :param threshold_fit: The maximum allowable R_fit
    :param threshold_walen: The Walen slope threshold for excluding Alfven waves.
    :param n_axis_iterations: Number of iterations for axis determination algorithm.
    Each iteration `i in {0, ..., n_axis_iterations - 1}` will scan a region of angular width $\pi/2^i$
    centered around the previous best orientation (using avg magnetic field direction as an initial guess)
    with n_trial_axes evenly spaced guess orientations.
    This feature is different from the original GS automated detection algorithm. Similar behavior can be obtained with `n_axis_iterations = 1`.
    :param n_trial_axes: The number of evenly spaced guess orientations used for the axis determination algorithm.
    Odd number is desirable so that one of the trial axes includes the previous iteration's best orientation (starting with avg field direction).
    :param max_axis_determination_resolution: Maximum size of samples used for axis determination algorithm residue calculation.
    Useful to limit since it makes it much faster to use less data for this intensive part of the algorithm.
    If the number of samples is greater than this amount, it is downsampled with adaptive average pooling.
    Can set to None to leave it as unlimited.
    :param min_positive_axial_component: Percentage of Bz to require to be positive. Default 0.95 (i.e. 95%).
    For behavior of original GS algorithm, set to 0.
    :param cuda: Whether to use the GPU
    :param axis_optimizer: Function of (magnetic_field, frames, vertical_directions) that returns optimized axes. Intended for deep learning based axis optimization.
    :return: A DataFrame describing the detected flux ropes.
    """

    tensor = _pack_data(magnetic_field, velocity, density, gas_pressure)

    # these are updated as the algorithm runs
    # contains_existing keeps track of samples that were already confirmed as MFRs
    # for longer durations so that shorter durations do not attempt to overwrite them
    contains_existing = torch.zeros(len(magnetic_field), dtype=torch.bool)
    remaining_events = list()

    sliding_windows = _get_sliding_windows(window_lengths, window_steps)

    for duration, window_step in tqdm.tqdm(sliding_windows):
        # dictionary: index of inflection point -> flux rope candidate
        event_candidates = dict()
        
        # generate the sliding windows
        windows = tensor.unfold(size=duration, step=window_step, dimension=0)
        window_starts = torch.arange(len(windows)) * window_step
        overlap_batches = contains_existing.unfold(size=duration, step=window_step, dimension=0)

        # iterate the windows of the same duration in batches to avoid running out of memory
        for i_batch in range(0, len(windows), batch_size):
            batch_starts = window_starts[i_batch:i_batch + batch_size]
            batch_data = windows[i_batch:i_batch + batch_size].transpose(1, 2)

            initial_mask = ~torch.any(torch.any(torch.isnan(batch_data), dim=2), dim=1) & \
                    (torch.norm(batch_data[:, :, :3], dim=2).mean(dim=1) >= min_strength) & \
                    ~torch.any(overlap_batches[i_batch:i_batch + batch_size], dim=1)
            
            batch_starts = batch_starts[initial_mask]
            batch_data = batch_data[initial_mask]

            if len(batch_data) == 0:
                continue
        
            if cuda:
                batch_data = batch_data.cuda()
                batch_starts = batch_starts.cuda()
            
            # extract individual physical quantities
            batch_magnetic_field = batch_data[:, :, :3]
            batch_velocity = batch_data[:, :, 3:6]
            batch_gas_pressure = batch_data[:, :, 9]

            batch_frames = _find_frames(batch_magnetic_field, batch_velocity, frame_type)
        
            max_clip = window_step

            batch_vertical_direction = _determine_vertical_directions(batch_magnetic_field,
                                                                      batch_frames,
                                                                      n_axis_iterations,
                                                                      n_trial_axes,
                                                                      max_axis_determination_resolution,
                                                                      max_clip,
                                                                      axis_optimizer,
                                                                      batch_size)
            
            batch_normalized_potential = _calculate_normalized_potential(batch_magnetic_field, batch_vertical_direction, batch_frames)

            inflection_point_mask = _count_inflection_points(batch_normalized_potential) == 1

            # estimate alfvenicity using walen slope
            alfvenicity_mask = _calculate_alfvenicity(batch_frames, batch_data) <= threshold_walen

            mask = inflection_point_mask & alfvenicity_mask
            
            if not torch.any(mask):
                continue

            batch_starts = batch_starts[mask]
            batch_data = batch_data[mask]
            batch_magnetic_field = batch_magnetic_field[mask]
            batch_velocity = batch_velocity[mask]
            batch_gas_pressure = batch_gas_pressure[mask]
            batch_frames = batch_frames[mask]
            batch_vertical_direction = batch_vertical_direction[mask]
            batch_normalized_potential = batch_normalized_potential[mask]

            # use the potential peaks as the inflection points
            inflection_points = batch_normalized_potential[..., 1:-1].abs().argmax(dim=1) + 1            
            
            batch_axes = _minimize_axis_residue(batch_magnetic_field, batch_vertical_direction, batch_frames, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, max_clip, axis_optimizer, return_residue=False)

            # rotate the magnetic field data into the flux rope coordinate system
            rotated_field = _rotate_field(batch_axes, batch_magnetic_field, batch_frames)
            
            # calculate $p_t = B_z^2/2mu_0 + p$
            transverse_magnetic_pressure = (rotated_field[:, :, 2] * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
            transverse_pressure = batch_gas_pressure + transverse_magnetic_pressure
            
            # require the inflection point pressure to be within the top 85%
            peak_pressure = transverse_pressure[torch.arange(len(transverse_pressure), device=batch_data.device), inflection_points]
            pressure_peak_mask = peak_pressure > torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")

            # require a certain percentage of Bz to be positive
            mask_positive_Bz = (rotated_field[:, :, 2] > 0).to(torch.float32).mean(dim=-1) > min_positive_axial_component

            # combine all the masks
            mask = pressure_peak_mask & mask_positive_Bz
            
            # calculate difference residue
            error_diff = torch.ones(len(mask), dtype=batch_data.dtype, device=batch_data.device) * np.nan
            error_diff[mask] = _calculate_residue_diff(inflection_points[mask], batch_normalized_potential[mask], transverse_pressure[mask])
            mask &= (error_diff <= threshold_diff)

            for i in torch.nonzero(mask).flatten():
                inflection_point = inflection_points[i]
                start = batch_starts[i].item()
                event_index = start + inflection_point.item()
                error_diff_i = error_diff[i].item()

                if event_index in event_candidates:
                    if error_diff_i >= event_candidates[event_index][0]:
                        continue

                error_fit = _calculate_residue_fit(batch_normalized_potential[i], transverse_pressure[i])

                start = start
                end = start + duration - 1
                event_candidates[event_index] = (error_diff_i,
                                                 start,
                                                 end,
                                                 duration,
                                                 *tuple(batch_axes[i].cpu().numpy()),
                                                 *tuple(batch_frames[i].cpu().numpy()),
                                                 error_fit.item())

        # done at the end of processing all batches with a given duration
        _cleanup_candidates(contains_existing, event_candidates, remaining_events, threshold_fit)

    remaining_events = list(sorted(remaining_events, key=lambda x: x[1]))
    return pd.DataFrame(remaining_events, columns=["error_diff", "start", "end", "duration",
                                                   "axis_x", "axis_y", "axis_z",
                                                   "frame_x", "frame_y", "frame_z",
                                                   "error_fit"])


def _find_frames(magnetic_field, velocity, frame_type):


    if frame_type == "mean_velocity":
        frames = velocity.mean(dim=1)
    elif frame_type == "vht":
        electric_field = -torch.cross(velocity, magnetic_field, dim=2)
        frames = estimate_ht_frame(magnetic_field, electric_field)
    else:
        raise Exception(f"Unknown frame type {frame_type}")
    return frames


def _minimize_axis_residue(magnetic_field, vertical_directions, frames, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, max_clip, axis_optimizer, return_residue=True):
    if axis_optimizer is not None:
        axes = axis_optimizer(magnetic_field, frames, vertical_directions)
        if return_residue:
            residue = calculate_residue_map(magnetic_field, frames, axes.unsqueeze(1), max_clip=max_clip).squeeze(-1)
    else:
        axes = F.normalize(magnetic_field.mean(dim=1), dim=-1)
        
        if max_axis_determination_resolution is not None and magnetic_field.shape[1] > max_axis_determination_resolution:
            magnetic_field_short = F.adaptive_avg_pool1d(magnetic_field.transpose(1, 2), max_axis_determination_resolution).transpose(1, 2)
        else:
            magnetic_field_short = magnetic_field
        
        for i in range(n_axis_iterations):
            spread = 1 / (2 ** i)
            axes, residue = _minimize_axis_residue_narrower(magnetic_field_short, vertical_directions, frames, n_trial_axes, max_clip, axes, spread)

    if return_residue:
        return axes, residue
    else:
        return axes


def _minimize_axis_residue_narrower(magnetic_field, vertical_directions, frames, n_trial_axes, max_clip, initial_axes, spread):
    # horizontal direction for temporary coordinate system (we want vertical direction up and one of the axes to be <B>)
    horizontal_directions = F.normalize(torch.cross(vertical_directions, initial_axes, dim=-1), dim=-1)

    # rotation matrix from flux rope coordinate to data coordinates
    rotation_matrix = torch.stack([horizontal_directions, vertical_directions, initial_axes], dim=-1)

    # generate trial axes by rotating <B> around the vertical axis (we expect z to be <B> +- 90 degrees around y, since <Bz> should be positive)
    phi = torch.linspace(-np.pi / 2 * spread, np.pi / 2 * spread, n_trial_axes, device=rotation_matrix.device, dtype=rotation_matrix.dtype)
    in_plane_vectors = torch.stack([torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)], dim=-1)
    in_plane_vectors = in_plane_vectors.view((*[1] * (len(vertical_directions.shape) - 1), *in_plane_vectors.shape)).expand((*vertical_directions.shape[:-1], *in_plane_vectors.shape))
    axes = (rotation_matrix @ in_plane_vectors.transpose(-1, -2)).transpose(-1, -2)

    # For determining the axis we only use magnetic field data, not gas pressure
    # This is because gas pressure error diff is not affected by rotations about the y axis
    # unlike Bz, which depends on the axis orientation
    # Bz itself is conserved along field lines, and this way we can compare the SIGN of Bz as well
    error_diff_axis = calculate_residue_map(magnetic_field, frames, axes, max_clip=max_clip)

    # select the best orientation
    residue, argmin = error_diff_axis.min(dim=-1)
    index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
    axes = axes.gather(-2, index).squeeze(-2)
    return axes, residue



def _determine_vertical_directions(magnetic_field, frames, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, max_clip, axis_optimizer, batch_size):
    # Implements a trick I came up with to narrow down the orientation possibilities--
    # essentially, we can determine the y direction analytically
    # because <\vec{B}> is perp to y since -int_0^1 By dx = A_f - A_0 = 0 and x \propto t
    # in addition to \vec{v}_{HT} being perp to y
    # since both are perp and (almost) always nonparallel, we can determine y
    # however, in some cases they are nearly parallel, which I will handle by performing trial and error on y
    # -H. Farooki
    
    # direction of spacecraft motion through flux rope axis and cross section
    path_direction = F.normalize(-frames)
    mean_magnetic_field_directions = F.normalize(magnetic_field.mean(dim=1), dim=-1)

    cross_product = torch.cross(mean_magnetic_field_directions, path_direction, dim=-1)

    vertical_directions = F.normalize(cross_product, dim=-1)

    # compare ratio of average along guessed vertical direction and the one perpendicular. if less than 10x difference, not well determined 
    perpendicular_direction = F.normalize(torch.cross(path_direction, vertical_directions))
    ratio = (magnetic_field * vertical_directions.unsqueeze(1)).sum(dim=2).mean(dim=1).abs() / (magnetic_field * perpendicular_direction.unsqueeze(1)).sum(dim=2).mean(dim=1).abs()
    too_close = (ratio > 0.1) | (torch.norm(vertical_directions, dim=-1) == 0)  # also too close if norm of vertical direction is 0

    if not torch.all(~too_close):
        # use pca to determine a guess y axis (useful if the non-y axis with average zero is a constant zero with maybe small noise)
        too_close_field = magnetic_field[too_close]
        too_close_mean_directions = mean_magnetic_field_directions[too_close]
        too_close_frames = frames[too_close]
        too_close_path_directions = path_direction[too_close]
        perpendicular_field = too_close_field - (too_close_path_directions.unsqueeze(-2) * too_close_field).sum(dim=-1, keepdims=True) * too_close_path_directions.unsqueeze(-2)
        principal_directions = torch.linalg.svd(perpendicular_field, full_matrices=False)[2][:, 0, :]
        assert torch.all(torch.abs((principal_directions * too_close_path_directions).sum(dim=-1)) < .001), "principal direction not perpendicular to velocity??"

        alternative_directions = F.normalize(torch.cross(principal_directions, too_close_path_directions, dim=-1))
        
        n_vertical_trials = 180  # 4 degree separation -> 45 vertical axis guesses. Odd number is desirable

        theta = torch.linspace(-np.pi / 2, np.pi / 2, n_vertical_trials, device=principal_directions.device).reshape(1, n_vertical_trials, 1)
        possible_vertical_directions = F.normalize(principal_directions.unsqueeze(1) * torch.cos(theta) + alternative_directions.unsqueeze(1) * torch.sin(theta), dim=-1)
        
        # ensure that x unit points opposite direction of frames
        x_unit = F.normalize(torch.cross(possible_vertical_directions, too_close_mean_directions.unsqueeze(1), dim=-1), dim=-1)
        x_unit_sign = -torch.sign((x_unit * too_close_frames.unsqueeze(1)).sum(dim=-1, keepdim=True))
        possible_vertical_directions *= x_unit_sign

        n_too_close = len(too_close_field)

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
            axis_batch_frames = torch.repeat_interleave(too_close_frames, axis_batch_size, dim=0)
            
            # looks like [axis1a, axis1b, axis1c, axis2a, axis2b, axis2c] 
            axis_batch_directions = possible_vertical_directions[:, i_axis_batch:i_axis_batch + axis_batch_size, :].reshape(-1, 3)
            
            axis_batch_potential = _calculate_normalized_potential(axis_batch_field,
                                                                   axis_batch_directions,
                                                                   axis_batch_frames)
            axis_batch_inflection_mask = _count_inflection_points(axis_batch_potential) == 1

            axis_batch_min_residue = torch.ones(len(axis_batch_directions),
                                                dtype=axis_batch_directions.dtype,
                                                device=axis_batch_directions.device) * torch.inf
            axis_batch_min_residue[axis_batch_inflection_mask] = _minimize_axis_residue(axis_batch_field[axis_batch_inflection_mask],
                                                    axis_batch_directions[axis_batch_inflection_mask],
                                                    axis_batch_frames[axis_batch_inflection_mask],
                                                    n_axis_iterations,
                                                    n_trial_axes,
                                                    max_axis_determination_resolution,
                                                    max_clip,
                                                    axis_optimizer)[1]
    

            # each entry of minimum_residues looks like
            # [[residue1a, residue1b, residue1c], [residue2a, residue2b, residue2c]]
            minimum_residues.append(axis_batch_min_residue.reshape(n_too_close, axis_batch_size))
        
        # looks like [[residue1a, ..., residue1z], [residue2a, ..., residue2z]]
        min_residue = torch.concat(minimum_residues, dim=1)
        assert min_residue.shape == (n_too_close, n_vertical_trials)
        argmin = min_residue.argmin(dim=-1)
        index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
        vertical_directions[too_close] = possible_vertical_directions.gather(-2, index).squeeze(-2)

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
    alfven_velocity = (magnetic_field[:, :3] / torch.sqrt(density.unsqueeze(1))).flatten(1) * 21.8114

    # combine needed data into one tensor
    return torch.concat([magnetic_field, velocity, alfven_velocity, gas_pressure], dim=1)


def _get_sliding_windows(window_lengths, window_steps):
    assert list(sorted(window_lengths)) == list(sorted(set(window_lengths))), "window lengths must be unique"
    assert len(window_lengths) == len(window_steps), "must have same number of window steps and lengths"

    # use sliding windows sorted by longest duration to shortest
    # because the algorithm gives preference to longer MFRs candidates over shorter ones
    # since long MFRs tend to show up as a bunch of small MFRs (especially ICMEs)
    return list(reversed(sorted(zip(window_lengths, window_steps), key=lambda x: x[0])))



def _cleanup_candidates(contains_existing, event_candidates, remaining_events, threshold_fit):
    # sort by end time
    event_candidates = dict(sorted(event_candidates.items(), reverse=False, key=lambda item: item[1][1]))
    for key in list(event_candidates.keys()):
        if key not in event_candidates:
            continue
        event = event_candidates[key]
        del event_candidates[key]

        error_fit = event[-1]

        start = event[1]
        end = event[2]
        duration = event[3]

        if error_fit > threshold_fit:
            continue

        remaining_events.append(event)
        contains_existing[start:start + duration] = True

        for other_key in list(event_candidates.keys()):
            ends_before = event_candidates[other_key][2] < start
            starts_after = event_candidates[other_key][1] > end
            if ends_before or starts_after:
                continue

            del event_candidates[other_key]


def _calculate_alfvenicity(frame, windows):
    remaining_flow = (windows[:, :, 3:6] - frame.unsqueeze(1)).squeeze(1).flatten(1)
    alfven_velocity = windows[:, :, 6:9].flatten(1)
    d_flow = remaining_flow - remaining_flow.mean(dim=1, keepdim=True)
    d_alfven = alfven_velocity - alfven_velocity.mean(dim=1, keepdim=True)
    walen_slope = (d_flow * d_alfven).sum(dim=1) / (d_alfven ** 2).sum(dim=1)
    return torch.abs(walen_slope)


def _count_inflection_points(potential):
    window_size = potential.shape[1] // 2
    if window_size % 2 == 0:
        window_size += 1
    filter = torch.as_tensor(savgol_coeffs(window_size, 3), dtype=potential.dtype, device=potential.device)
    smoothed = torch.conv1d(potential.unsqueeze(1), filter[None, None, :], padding="valid").squeeze(1)
    # TODO: Fix this padding!

    is_sign_change = (torch.diff(torch.sign(torch.diff(smoothed))) != 0).to(int)

    return torch.sum(is_sign_change, dim=-1)


def _calculate_normalized_potential(batch_magnetic_field, vertical_directions, batch_frames):
    # calculate the A array/magnetic flux function/potential using $A(x) = -\int_{x_0}^x B_y dx$
    batch_magnetic_field_y = torch.sum(batch_magnetic_field * vertical_directions.unsqueeze(-2), dim=-1)
    unscaled_potential = torch.zeros(batch_magnetic_field_y.shape, device=batch_magnetic_field_y.device)
    unscaled_potential[:, 1:] = torch.cumulative_trapezoid(batch_magnetic_field_y)
    normalized_potential = unscaled_potential / unscaled_potential.abs().amax(dim=-1, keepdim=True)
    return normalized_potential

