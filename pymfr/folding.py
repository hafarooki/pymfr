import torch
import torch.nn.functional as F


def _find_inflection_points(potential):
    duration = potential.shape[1]
    kernel_size = duration // 4 // 2 * 2 + 1  # divide by 4, floor, round up to nearest odd

    if kernel_size > 1:
        smoothed = F.avg_pool1d(potential,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=(kernel_size - 1) // 2,
                                count_include_pad=False)
        assert smoothed.shape[1] == duration
    else:
        smoothed = potential

    points = (torch.diff(torch.sign(torch.diff(smoothed))) != 0).int()

    inflection_points = points.argmax(dim=1) + 1
    inflection_point_counts = points.sum(dim=1)
    return inflection_point_counts, inflection_points


def _calculate_folding_mask(inflection_points, inflection_point_counts, transverse_pressure):
    peaks = transverse_pressure[torch.arange(len(transverse_pressure), device=transverse_pressure.device),
                                inflection_points]
    min_pressure, max_pressure = torch.aminmax(transverse_pressure, dim=1)
    thresholds = torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")
    mask = (inflection_point_counts == 1) & (peaks > thresholds) & ((max_pressure - min_pressure) > 0)
    return mask


def _calculate_trim_mask(potential, threshold_folding):
    trim_mask = (potential[:, -1].abs() / potential.abs().amax(dim=1)) < threshold_folding
    return trim_mask
