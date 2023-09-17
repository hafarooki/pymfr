import torch
from torchinterp1d import interp1d
import numpy as np


def _calculate_interpolated_values(potential, quantity):
    assert len(potential.shape) == 2
    assert len(quantity.shape) == 2

    inflection_points = potential[..., 1:-1].abs().argmax(dim=-1) + 1

    if len(inflection_points) == 0:
        return torch.empty((0,), device=inflection_points.device, dtype=torch.float32)

    # force to be positive at peak
    potential = torch.where(potential.gather(1, inflection_points.unsqueeze(1)) < 0, -potential, potential)

    # second dim is potential1, quantity1, potential2, quantity2
    folded_data = torch.concat((potential.unsqueeze(1), quantity.unsqueeze(1)), dim=1).repeat(1, 2, 1)
    duration = folded_data.shape[2]

    all_indices = torch.arange(duration, device=inflection_points.device)
    all_indices = all_indices.unsqueeze(0).expand(folded_data.shape[0], -1)
    before = all_indices < inflection_points.unsqueeze(1)
    after = all_indices > inflection_points.unsqueeze(1)

    indices = torch.arange(len(folded_data), device=folded_data.device)
    inflection_point_values = folded_data[indices, :2, inflection_points]
    inflection_point_values = inflection_point_values.unsqueeze(2).expand(-1, -1, duration)

    # in first half, values after inflection point should be the inflection points repeated
    folded_data[:, :2, :] = torch.where(after.unsqueeze(1), inflection_point_values, folded_data[:, :2, :])

    # in second half, values before inflection point should be the inflection points repeated
    folded_data[:, 2:, :] = torch.where(before.unsqueeze(1), inflection_point_values, folded_data[:, 2:, :])

    sort1 = torch.argsort(folded_data[:, 0, :], dim=1)
    folded_data[:, 0, :] = folded_data[:, 0, :].gather(1, sort1)
    folded_data[:, 1, :] = folded_data[:, 1, :].gather(1, sort1)

    sort2 = torch.argsort(folded_data[:, 2, :], dim=1)
    folded_data[:, 2, :] = folded_data[:, 2, :].gather(1, sort2)
    folded_data[:, 3, :] = folded_data[:, 3, :].gather(1, sort2)

    # scale x axis from ~0 to 1
    peak_values = folded_data[:, 2, :].amax(dim=-1, keepdim=True)
    folded_data[:, 0, :] /= peak_values
    folded_data[:, 2, :] /= peak_values

    # residue can be calculated either by interpolating both arrays onto a fixed set of points,
    # as in Hu & Sonnerup (2002), or by 
    # comparing each point to the corresponding interpolated point from the other branch,
    # as done in Hu et al. 2018.
    # The second approach seems better for detecting flux ropes because generally the edge of a
    # flux rope candidate has a steep gradient (large By) compared to the center,
    # so evenly spaced A values are biased towards the measurements away from the center
    # and over-use the interpolated values
    interpolated1 = interp1d(folded_data[:, 0, :],
                                   folded_data[:, 1, :],
                                   potential)
    interpolated2 = interp1d(folded_data[:, 2, :],
                                   folded_data[:, 3, :],
                                   potential)
    return interpolated1, interpolated2


def _calculate_folding_differences(potential, quantity):
    interpolated1, interpolated2 = _calculate_interpolated_values(potential, quantity)

    return interpolated2 - interpolated1


def _calculate_residue_diff(potential, field_line_invariant):
    assert len(potential.shape) == 2
    assert len(field_line_invariant.shape) == 2

    # because aminmax requires non empty tensor
    if len(potential) == 0:
        return torch.empty(0, device=potential.device, dtype=potential.dtype)
    
    interp_diff = _calculate_folding_differences(potential, field_line_invariant)

    min_field_line_invariant, max_field_line_invariant = torch.aminmax(field_line_invariant, dim=1)
    field_line_invariant_range = max_field_line_invariant - min_field_line_invariant
    # error_diff = torch.sqrt(torch.mean(interp_diff ** 2, dim=1)) / field_line_invariant_range
    error_diff = torch.sqrt(torch.mean(interp_diff ** 2, dim=1)) / field_line_invariant_range

    # infinity if 0 field_line_invariant range
    error_diff = torch.where(field_line_invariant_range > 0, error_diff, torch.inf)

    return error_diff
