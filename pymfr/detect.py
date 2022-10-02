import numpy as np
import torch
import tqdm as tqdm

from pymfr.axis import _get_trial_axes, _rotate_field, calculate_residue_map
from pymfr.frame import estimate_ht_frame, estimate_ht2d_frame

from pymfr.residue import _calculate_residue_fit


def detect_flux_ropes(magnetic_field,
                      velocity,
                      density,
                      gas_pressure,
                      batch_size_mb,
                      window_lengths,
                      window_steps,
                      min_strength,
                      frame_type="vht_2d",
                      altitude_range=range(10, 90, 5),
                      azimuth_range=range(0, 360, 5),
                      threshold_diff=0.12,
                      threshold_fit=0.14,
                      threshold_walen=0.3,
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
    - "vht_2d":
        Finds a frame with an approximately constant electric field magnitude
        along the trial axis, which in theory should be present for a 2D structure.
    :param altitude_range: Range of trial axis altitudes.
    :param azimuth_range: Range of trial axis azimuths.
    :param threshold_diff: The maximum allowable R_diff
    :param threshold_fit: The maximum allowable R_fit
    :param threshold_walen: The Walen slope threshold for excluding Alfven waves.
    0 would mean it has to be 0, 1 would mean it can be equal to the maximum value.
    This threshold is unique to this implementation--the original paper trims the window instead,
    which is not trivial to do with the efficient vectorized computations of this implementation.
    The threshold should be close enough to 0 so that we satisfy the assumption that
    we crossed the flux boundary twice, high enough to account for uncertainties in the measurements
    and calculations. By default, it is 0.5.
    :param cuda: Whether to use the GPU
    :return: A list of tuples.
    In the future this should be replaced with a list of specialized objects or a dataframe.
    """

    tensor = _pack_data(magnetic_field, velocity, density, gas_pressure)

    trial_axes = _get_trial_axes(altitude_range, azimuth_range)
    if cuda:
        trial_axes = trial_axes.cuda()

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

        window_size_mb = np.product(windows.shape[1:]) * len(trial_axes) * 32 / 1024 / 1024

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

            batch_frames, batch_axes = _find_frames(batch_data[:, :, :3], batch_data[:, :, 3:6], trial_axes, frame_type)

            error_diff = calculate_residue_map(batch_data[:, :, :3], batch_data[:, :, 9], batch_frames, batch_axes,
                                               max_clip=window_step)

            error_diff, argmin = error_diff.min(dim=-1)
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

            inflection_points, inflection_point_counts = _find_inflection_points(potential)

            peaks = transverse_pressure[torch.arange(len(transverse_pressure), device=transverse_pressure.device),
                                        inflection_points]
            thresholds = torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")
            folding_mask = (inflection_point_counts == 1) & \
                           (peaks > thresholds)

            mask = alfvenicity_mask & folding_mask & (error_diff <= threshold_diff)

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
                                                 tuple(batch_axes[i].cpu().numpy()),
                                                 tuple(batch_frames[i].cpu().numpy()),
                                                 error_fit.item())

            if cuda:
                torch.cuda.empty_cache()

        _cleanup_candidates(contains_existing, event_candidates, remaining_events, threshold_fit)

    return list(sorted(remaining_events, key=lambda x: x[1]))


def _find_frames(magnetic_field, velocity, trial_axes, frame_type):
    n_batch = len(magnetic_field)
    batch_axes = trial_axes.unsqueeze(0).expand(n_batch, -1, -1)

    electric_field = -torch.cross(velocity, magnetic_field, dim=2)

    n_trial_axis = len(trial_axes)

    if frame_type == "mean_velocity":
        batch_frames = velocity.mean(dim=1).unsqueeze(1).expand(-1, n_trial_axis, -1)
    elif frame_type == "vht_2d":
        batch_frames = estimate_ht2d_frame(magnetic_field, electric_field, batch_axes)
    elif frame_type == "vht":
        vht = estimate_ht_frame(magnetic_field, electric_field)
        batch_frames = vht.unsqueeze(1).expand(-1, n_trial_axis, -1)
    else:
        raise Exception(f"Unknown frame type {frame_type}")

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


def _find_inflection_points(potential):
    inflection_points = potential[..., 1:-1].abs().argmax(dim=1) + 1

    smoothed = _smooth(potential)

    points = (torch.diff(torch.sign(torch.diff(smoothed))) != 0).int()

    inflection_point_counts = points.sum(dim=1).long()
    return inflection_points, inflection_point_counts


def _smooth(potential):
    duration = potential.shape[1]
    kernel_size = duration // 16 // 2 * 2 + 1  # divide by 16, floor, round up to nearest odd
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


def _calculate_trim_mask(potential, threshold_folding):
    trim_mask = (potential[:, -1].abs() / potential.abs().amax(dim=1)) < threshold_folding
    return trim_mask
