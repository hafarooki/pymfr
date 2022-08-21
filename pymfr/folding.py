import torch
import torch.nn.functional as F


def _find_inflection_points(potential):
    inflection_points = potential[..., 1:-1].abs().argmax(dim=1) + 1

    smoothed = _smooth(potential)

    points = (torch.diff(torch.sign(torch.diff(smoothed))) != 0).int()

    inflection_point_counts = points.sum(dim=1).long()
    return inflection_points, inflection_point_counts


def _find_single_inflection_points(potential):
    return potential[..., 1:-1].abs().argmax(dim=-1) + 1


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


def _calculate_folding_mask(inflection_points, inflection_point_counts, transverse_pressure,
                            potential, window_step):
    peaks = transverse_pressure[torch.arange(len(transverse_pressure), device=transverse_pressure.device),
                                inflection_points]

    # force to be positive at peak
    potential = torch.where(potential.gather(1, inflection_points.unsqueeze(1)) < 0, -potential, potential)
    min_values = torch.clamp_min(potential[..., -1], 0)
    trim_count = (potential < min_values.unsqueeze(1)).long().sum(dim=1)

    min_pressure, max_pressure = torch.aminmax(transverse_pressure, dim=1)
    thresholds = torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")
    mask = (trim_count <= window_step) & \
           (inflection_point_counts == 1) & \
           (peaks > thresholds) & \
           (max_pressure - min_pressure > 0)
    return mask


def _calculate_trim_mask(potential, threshold_folding):
    trim_mask = (potential[:, -1].abs() / potential.abs().amax(dim=1)) < threshold_folding
    return trim_mask
