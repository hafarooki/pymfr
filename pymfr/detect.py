import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm as tqdm

from pymfr.axis import _get_trial_axes, _rotate_field, calculate_residue_map
from pymfr.frame import estimate_ht_frame

from pymfr.residue import _calculate_residue_diff, _calculate_residue_fit


def detect_flux_ropes(magnetic_field,
                      velocity,
                      density,
                      gas_pressure,
                      batch_size_mb,
                      window_lengths,
                      window_steps,
                      min_strength,
                      frame_type="vht",
                      smooth_factor=16,
                      threshold_diff=0.12,
                      threshold_fit=0.14,
                      threshold_walen=0.3,
                      n_trial_axes=128,
                      min_positive_axial_component=0.95,
                      cuda=True):
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
    :param batch_size_mb: Maximum batch size in megabytes.
    This is to set an approximate limit to the GPU memory usage. In reality the memory usage
    should exceed this amount, so it should be set as high as it can be without the GPU running out
    of memory for optimal performance.
    :param window_lengths: A range of sliding window sizes.
    :param window_steps: A range of sliding window strides.
    :param min_strength: Minimum <B> for sliding window candidates in nT.
    :param frame_type: The method used to estimate the propagation direction of the flux rope.
    Options:
    - "mean_velocity"
        Simply takes the average velocity over the window. Fastest but least precise.
    - "vht":
        Finds the frame that minimizes the averaged electric field magnitude squared.
    :param smooth_factor: The A array is smoothed by duration//smooth_factor moving average before checking for multiple inflection points
    :param threshold_diff: The maximum allowable R_diff
    :param threshold_fit: The maximum allowable R_fit
    :param threshold_walen: The Walen slope threshold for excluding Alfven waves.
    :param require_positive_axial_component: Percentage of Bz to require to be positive. Default 0.95 (i.e. 95%).
    0 would mean it has to be 0, 1 would mean it can be equal to the maximum value.
    This threshold is unique to this implementation--the original paper trims the window instead,
    which is not trivial to do with the efficient vectorized computations of this implementation.
    The threshold should be close enough to 0 so that we satisfy the assumption that
    we crossed the flux boundary twice, high enough to account for uncertainties in the measurements
    and calculations. By default, it is 0.5.
    :param cuda: Whether to use the GPU
    :return: A DataFrame describing the detected flux ropes.
    """

    tensor = _pack_data(magnetic_field, velocity, density, gas_pressure)

    # these are updated as the algorithm runs
    # contains_existing keeps track of samples that were already confirmed as MFRs
    # for longer durations so that shorter durations do not attempt to overwrite them
    contains_existing = torch.zeros(len(magnetic_field), dtype=torch.bool)
    remaining_events = []

    sliding_windows = _get_sliding_windows(window_lengths, window_steps)

    for duration, window_step in tqdm.tqdm(sliding_windows):
        event_candidates = {}

        windows = tensor.unfold(size=duration, step=window_step, dimension=0).transpose(1, 2)
        window_starts = torch.arange(len(windows)) * window_step
        overlap_batches = contains_existing.unfold(size=duration, step=window_step, dimension=0)

        window_size_mb = np.product(windows.shape[1:]) * 32 / 1024 / 1024

        assert batch_size_mb >= window_size_mb, f"Batch size ({batch_size_mb} MB) < window size ({window_size_mb} MB)"
        batch_size = int(max(batch_size_mb / window_size_mb // 8 * 8, 1))

        for i_batch in reversed(range(0, len(windows), batch_size)):
            batch_data = windows[i_batch:i_batch + batch_size]
            batch_starts = window_starts[i_batch:i_batch + batch_size]
            batch_overlaps = overlap_batches[i_batch:i_batch + batch_size]

            if cuda:
                batch_data = batch_data.cuda()
                batch_starts = batch_starts.cuda()
                batch_overlaps = batch_overlaps.cuda()

            mask = ~torch.any(torch.any(torch.isnan(batch_data), dim=2), dim=1) & \
                   (torch.norm(batch_data[:, :, :3], dim=2).mean(dim=1) >= min_strength) & \
                   ~torch.any(batch_overlaps, dim=1)

            batch_data = batch_data[mask]
            batch_starts = batch_starts[mask]

            if len(batch_data) == 0:
                continue

            batch_frames, batch_axes = _find_frames(batch_data[:, :, :3], batch_data[:, :, 3:6], n_trial_axes, frame_type)

            # For determining the axis we only use magnetic field data, not gas pressure
            # This is because gas pressure error diff is not affected by rotations about the y axis
            # unlike Bz, which depends on the axis orientation
            # Bz itself is conserved along field lines, and this way we can compare the SIGN of Bz as well
            error_diff_axis = calculate_residue_map(batch_data[:, :, :3], batch_frames, batch_axes,
                                               max_clip=window_step)

            error_diff_axis, argmin = error_diff_axis.min(dim=-1)
            index = argmin.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)
            batch_axes = batch_axes.gather(-2, index).squeeze(-2)
            batch_frames = batch_frames.gather(-2, index).squeeze(-2)

            alfvenicity = _calculate_alfvenicity(batch_frames, batch_data)

            rotated_field = _rotate_field(batch_axes, batch_data[:, :, :3], batch_frames)

            potential = torch.zeros((len(rotated_field)), rotated_field.shape[1], device=rotated_field.device)
            potential[:, 1:] = torch.cumulative_trapezoid(rotated_field[:, :, 1])

            batch_gas_pressure = batch_data[:, :, 9]
            transverse_magnetic_pressure = (rotated_field[:, :, 2] * 1e-9) ** 2 / (2 * 1.25663706212e-6) * 1e9
            transverse_pressure = batch_gas_pressure + transverse_magnetic_pressure

            alfvenicity_mask = alfvenicity <= threshold_walen

            inflection_points, inflection_point_counts = _find_inflection_points(potential, smooth_factor)

            peaks = transverse_pressure[torch.arange(len(transverse_pressure), device=transverse_pressure.device),
                                        inflection_points]
            thresholds = torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")
            folding_mask = (inflection_point_counts == 1) & \
                           (peaks > thresholds)
            error_diff = _calculate_residue_diff(inflection_points, potential, transverse_pressure)

            mask_positive_Bz = (rotated_field[:, :, 2] > 0).to(torch.float32).mean(dim=-1) > min_positive_axial_component

            mask = alfvenicity_mask & folding_mask & (error_diff <= threshold_diff) & mask_positive_Bz

            for i in torch.nonzero(mask).flatten():
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

            if cuda:
                torch.cuda.empty_cache()

        _cleanup_candidates(contains_existing, event_candidates, remaining_events, threshold_fit)

    remaining_events = list(sorted(remaining_events, key=lambda x: x[1]))
    return pd.DataFrame(remaining_events, columns=["error_diff", "start", "end", "duration",
                                                   "axis_x", "axis_y", "axis_z",
                                                   "frame_x", "frame_y", "frame_z",
                                                   "error_fit"])


def _find_frames(magnetic_field, velocity, n_trial_axes, frame_type):
    n_batch = len(magnetic_field)

    # Implements a trick I came up with and presented at NRSM 2023--
    # essentially, we can determine the y direction analytically
    # because <\vec{B}> is perp to y since -int_0^1 By dx = A_f - A_0 = 0 and x \propto t
    # in addition to \vec{v}_{HT} being perp to y
    # since both are perp and (almost) always nonparallel, we can determine y! isn't it fun?
    # -H. Farooki

    if frame_type == "mean_velocity":
        frames = velocity.mean(dim=1)
    elif frame_type == "vht":
        electric_field = -torch.cross(velocity, magnetic_field, dim=2)
        frames = estimate_ht_frame(magnetic_field, electric_field)
    else:
        raise Exception(f"Unknown frame type {frame_type}")
    
    mean_magnetic_field_directions = F.normalize(magnetic_field.mean(dim=1), dim=-1)

    # this is the y direction
    vertical_direction = F.normalize(torch.cross(mean_magnetic_field_directions, -frames, dim=-1), dim=-1)

    # horizontal direction assuming z is along <B>
    horizontal_direction = F.normalize(torch.cross(vertical_direction, mean_magnetic_field_directions, dim=-1), dim=-1)

    # rotation matrix from flux rope coordinate to data coordinates
    rotation_matrix = torch.stack([horizontal_direction, vertical_direction, mean_magnetic_field_directions], dim=2)
    theta = torch.linspace(-np.pi / 2, np.pi / 2, n_trial_axes, device=rotation_matrix.device)
    in_plane_vectors = torch.stack([torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], dim=-1)
    in_plane_vectors = in_plane_vectors.unsqueeze(0).expand(n_batch, -1, -1)
    batch_axes = (rotation_matrix @ in_plane_vectors.transpose(1, 2)).transpose(1, 2)

    batch_frames = frames.unsqueeze(1).expand(-1, n_trial_axes, -1)

    return batch_frames, batch_axes


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
    magnetic_field = torch.as_tensor(magnetic_field)
    velocity = torch.as_tensor(velocity)
    density = torch.as_tensor(density)
    gas_pressure = torch.as_tensor(gas_pressure).unsqueeze(1)

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


def _resize(batch_data, batch_starts, trial_axes):
    batch_data = batch_data.repeat(len(trial_axes), 1, 1)
    batch_starts = batch_starts.repeat(len(trial_axes))
    return batch_data, batch_starts


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


def _find_inflection_points(potential, smooth_factor):
    inflection_points = potential[..., 1:-1].abs().argmax(dim=1) + 1

    smoothed = _smooth(potential, smooth_factor)

    points = (torch.diff(torch.sign(torch.diff(smoothed))) != 0).int()

    inflection_point_counts = points.sum(dim=1).long()
    return inflection_points, inflection_point_counts


def _smooth(potential, smooth_factor):
    duration = potential.shape[1]
    kernel_size = duration // smooth_factor // 2 * 2 + 1  # divide by smooth_factor, floor, round up to nearest odd
    if kernel_size > 1:
        smoothed = F.avg_pool1d(potential,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=(kernel_size - 1) // 2,
                                count_include_pad=False)
        assert smoothed.shape[1] == duration
    else:
        smoothed = potential
    return smoothed
\