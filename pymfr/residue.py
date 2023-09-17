import torch
from torchinterp1d import interp1d
import numpy as np


def _calculate_interpolated_values(potential, quantity):
    assert len(potential.shape) == 2
    assert len(quantity.shape) == 2

    if len(potential) == 0:
        return torch.empty((0,), device=potential.device, dtype=torch.float32)

    inflection_points = potential[..., 1:-1].abs().argmax(dim=-1) + 1

    # fix peak at 1
    potential = potential / potential.gather(-1, inflection_points.unsqueeze(1))

    duration = potential.shape[-1]

    all_indices = torch.arange(duration, device=inflection_points.device)
    all_indices = all_indices.unsqueeze(0).expand(potential.shape[0], -1)
    before = all_indices < inflection_points.unsqueeze(1)
    after = all_indices > inflection_points.unsqueeze(1)

    inflection_point_quantity = quantity.gather(-1, inflection_points.unsqueeze(1)).squeeze(1)

    # in first half, values after inflection point should be the inflection points repeated
    potential_left = torch.where(after, 1, potential)
    quantity_left = torch.where(after, inflection_point_quantity.unsqueeze(1), quantity)

    # in second half, values before inflection point should be the inflection points repeated
    potential_right = torch.where(before, 1, potential)
    quantity_right = torch.where(before, inflection_point_quantity.unsqueeze(1), quantity)

    potential_left, argsort_left = torch.sort(potential_left, dim=1)
    quantity_left = quantity_left.gather(1, argsort_left)

    potential_right, argsort_right = torch.sort(potential_right, dim=1)
    quantity_right = quantity_right.gather(1, argsort_right)

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
