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


def _calculate_trim_mask(potential, threshold_folding):
    trim_mask = (potential[:, -1].abs() / potential.abs().amax(dim=1)) < threshold_folding
    return trim_mask
