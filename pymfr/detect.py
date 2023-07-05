import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm as tqdm

from scipy.signal import savgol_filter

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

        # generate the sliding windows, and transpose to make it so the dimensions are (batch, time, physical quantity)
        windows = tensor.unfold(size=duration, step=window_step, dimension=0).transpose(1, 2)
        window_starts = torch.arange(len(windows)) * window_step
        overlap_batches = contains_existing.unfold(size=duration, step=window_step, dimension=0)

        # do not process windows that have missing data, do not fit the min field strength, or overlap with an existing flux rope candidate of longer duration
        # we these windows from the start to improve performance
        mask = ~torch.any(torch.any(torch.isnan(windows), dim=2), dim=1) & \
                (torch.norm(windows[:, :, :3], dim=2).mean(dim=1) >= min_strength) & \
                ~torch.any(overlap_batches, dim=1)

        windows = windows[mask]
        window_starts = window_starts[mask]

        # skip if no windows remain after applying the mask
        if len(windows) == 0:
            continue

        if cuda:
            windows.pin_memory("cuda")
            window_starts.pin_memory("cuda")

        # iterate the windows of the same duration in batches to avoid running out of memory
        for i_batch in reversed(range(0, len(windows), batch_size)):
            batch_data = windows[i_batch:i_batch + batch_size]
            batch_starts = window_starts[i_batch:i_batch + batch_size]

            if cuda:
                batch_data = batch_data.cuda(non_blocking=True)
                batch_starts = batch_starts.cuda(non_blocking=True)
            
            # extract individual physical quantities
            batch_magnetic_field = batch_data[:, :, :3]
            batch_velocity = batch_data[:, :, 3:6]
            batch_gas_pressure = batch_data[:, :, 9]

            # first find the velocity frames and the minimum difference residue axes
            batch_frames, batch_axes = _find_frames(batch_magnetic_field,
                                                    batch_velocity,
                                                    n_axis_iterations,
                                                    n_trial_axes,
                                                    max_axis_determination_resolution,
                                                    frame_type,
                                                    max_clip=window_step,
                                                    axis_optimizer=axis_optimizer)

            # estimate alfvenicity using walen slope
            alfvenicity = _calculate_alfvenicity(batch_frames, batch_data)
            alfvenicity_mask = alfvenicity <= threshold_walen

            # rotate the magnetic field data into the flux rope coordinate system
            rotated_field = _rotate_field(batch_axes, batch_magnetic_field, batch_frames)
            
            # calculate the A array/magnetic flux function/potential using $A(x) = -\int_{x_0}^x B_y dx$
            potential = torch.zeros((len(rotated_field)), rotated_field.shape[1], device=rotated_field.device)
            potential[:, 1:] = torch.cumulative_trapezoid(rotated_field[:, :, 1])

            # use the potential peaks as the inflection points
            inflection_points = potential[..., 1:-1].abs().argmax(dim=1) + 1
            
            # calculate $p_t = B_z^2/2mu_0 + p$
            transverse_magnetic_pressure = (rotated_field[:, :, 2] * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
            transverse_pressure = batch_gas_pressure + transverse_magnetic_pressure
            
            # require the inflection point pressure to be within the top 85%
            peak_pressure = transverse_pressure[torch.arange(len(transverse_pressure), device=batch_data.device), inflection_points]
            pressure_peak_mask = peak_pressure > torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")

            # require a certain percentage of Bz to be positive
            mask_positive_Bz = (rotated_field[:, :, 2] > 0).to(torch.float32).mean(dim=-1) > min_positive_axial_component

            # combine all the masks
            mask = alfvenicity_mask & pressure_peak_mask & mask_positive_Bz
            
            # calculate difference residue
            error_diff = torch.ones(len(mask), dtype=batch_data.dtype, device=batch_data.device) * np.nan
            error_diff[mask] = _calculate_residue_diff(inflection_points[mask], potential[mask], transverse_pressure[mask])
            mask &= (error_diff <= threshold_diff)

            for i in torch.nonzero(mask).flatten():
                if _count_inflection_points(potential[i]) > 1:
                    continue

                inflection_point = inflection_points[i]
                start = batch_starts[i]
                event_index = start + inflection_point.item()
                error_diff_i = error_diff[i].item()

                if event_index in event_candidates:
                    if error_diff_i >= event_candidates[event_index][0]:
                        continue

                error_fit = _calculate_residue_fit(potential[i], transverse_pressure[i])

                start = start
                end = start + duration - 1
                event_candidates[event_index] = (error_diff_i,
                                                 start.item(),
                                                 end.item(),
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


def _find_frames(magnetic_field, velocity, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, frame_type, max_clip, axis_optimizer):
    # Implements a trick I came up with to narrow down the orientation possibilities--
    # essentially, we can determine the y direction analytically
    # because <\vec{B}> is perp to y since -int_0^1 By dx = A_f - A_0 = 0 and x \propto t
    # in addition to \vec{v}_{HT} being perp to y
    # since both are perp and (almost) always nonparallel, we can determine y
    # however, in some cases they are nearly parallel, which I will handle by performing trial and error on y
    # -H. Farooki

    if frame_type == "mean_velocity":
        frames = velocity.mean(dim=1)
    elif frame_type == "vht":
        electric_field = -torch.cross(velocity, magnetic_field, dim=2)
        frames = estimate_ht_frame(magnetic_field, electric_field)
    else:
        raise Exception(f"Unknown frame type {frame_type}")

    vertical_directions = _determine_vertical_directions(magnetic_field, frames, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, max_clip, axis_optimizer)

    axes = _minimize_axis_residue(magnetic_field, vertical_directions, frames, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, max_clip, axis_optimizer, return_residue=False)

    return frames, axes


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



def _determine_vertical_directions(magnetic_field, frames, n_axis_iterations, n_trial_axes, max_axis_determination_resolution, max_clip, axis_optimizer):
    # direction of spacecraft motion through flux rope axis and cross section
    path_direction = F.normalize(-frames)
    mean_magnetic_field_directions = F.normalize(magnetic_field.mean(dim=1), dim=-1)

    cross_product = torch.cross(mean_magnetic_field_directions, path_direction, dim=-1)

    vertical_direction = F.normalize(cross_product, dim=-1)

    # compare ratio of average along guessed vertical direction and the one perpendicular. if less than 10x difference, not well determined 
    perpendicular_direction = F.normalize(torch.cross(path_direction, vertical_direction))
    ratio = (magnetic_field * vertical_direction.unsqueeze(1)).sum(dim=2).mean(dim=1).abs() / (magnetic_field * perpendicular_direction.unsqueeze(1)).sum(dim=2).mean(dim=1).abs()
    too_close = (ratio > 0.1) | (torch.norm(vertical_direction, dim=-1) == 0)  # also too close if norm of vertical direction is 0
    

    if not torch.all(~too_close):
        # use pca to determine a guess y axis (useful if the non-y axis with average zero is a constant zero with maybe small noise)
        too_close_field = magnetic_field[too_close]
        too_close_frames = frames[too_close]
        too_close_directions = path_direction[too_close]
        perpendicular_field = too_close_field - (too_close_directions.unsqueeze(-2) * too_close_field).sum(dim=-1, keepdims=True) * too_close_directions.unsqueeze(-2)
        principal_directions = torch.linalg.svd(perpendicular_field, full_matrices=False)[2][:, 0, :]
        assert torch.all(torch.abs((principal_directions * too_close_directions).sum(dim=-1)) < .001), "principal direction not perpendicular to velocity??"

        alternative_directions = F.normalize(torch.cross(principal_directions, too_close_directions, dim=-1))
        
        n_vertical_trials = 180 // 4  # 4 degree separation -> 45 vertical axis guesses. Odd number is desirable

        theta = torch.linspace(-np.pi / 2, np.pi / 2, n_vertical_trials, device=principal_directions.device).reshape(1, n_vertical_trials, 1)
        possible_vertical_directions = principal_directions.unsqueeze(1) * torch.cos(theta) + alternative_directions.unsqueeze(1) * torch.sin(theta)
        
        # use for loop to avoid running out of memory with a large batch size
        min_residue = torch.stack([_minimize_axis_residue(too_close_field,
                                                    possible_vertical_directions[:, i, :],
                                                    too_close_frames,
                                                    n_axis_iterations,
                                                    n_trial_axes,
                                                    max_axis_determination_resolution,
                                                    max_clip,
                                                    axis_optimizer)[1] for i in range(n_vertical_trials)], dim=-1)
        _, argmin = min_residue.min(dim=-1)
        index = argmin.view((*argmin.shape, 1, 1)).expand((*argmin.shape, 1, 3))
        vertical_direction[too_close] = possible_vertical_directions.gather(-2, index).squeeze(-2)

    return vertical_direction


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
    window_size = len(potential) // 2
    if window_size % 2 == 0:
        window_size += 1
    smoothed = savgol_filter(potential.cpu().numpy(), window_length=window_size, polyorder=3)

    is_sign_change = (np.diff(np.sign(np.diff(smoothed))) != 0).astype(int)
    is_sign_change[0] = 0
    is_sign_change[-1] = 0

    return int(np.sum(is_sign_change))
