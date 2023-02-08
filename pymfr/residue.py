import torch
from torchinterp1d import Interp1d
import numpy as np


def _calculate_residue_diff(inflection_points, potential, field_line_invariant, max_clip=None):
    assert len(inflection_points.shape) == 1
    assert len(potential.shape) == 2
    assert len(field_line_invariant.shape) == 2

    if len(inflection_points) == 0:
        return torch.empty((0,), device=inflection_points.device, dtype=torch.float32)

    # force to be positive at peak
    potential = torch.where(potential.gather(1, inflection_points.unsqueeze(1)) < 0, -potential, potential)

    # second dim is potential1, field_line_invariant1, potential2, field_line_invariant2
    folded_data = torch.concat((potential.unsqueeze(1), field_line_invariant.unsqueeze(1)), dim=1).repeat(1, 2, 1)
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

    # scale x axis from 0 to 1
    minimum_values = torch.clamp(folded_data[:, 2, :].amin(dim=1, keepdim=True), min=0)
    unclipped_mask = potential >= minimum_values

    folded_data[:, 0, :] -= minimum_values
    folded_data[:, 2, :] -= minimum_values
    peak_values = folded_data[:, 2, :].amax(dim=-1, keepdim=True)
    folded_data[:, 0, :] /= peak_values
    folded_data[:, 2, :] /= peak_values

    potential_interp = torch.linspace(0, 1, duration, device=folded_data.device)
    potential_interp = potential_interp.unsqueeze(0).expand(folded_data.shape[0], -1)
    interpolated1 = Interp1d.apply(folded_data[:, 0, :],
                                   folded_data[:, 1, :],
                                   potential_interp)
    interpolated2 = Interp1d.apply(folded_data[:, 2, :],
                                   folded_data[:, 3, :],
                                   potential_interp)
    interp_diff = (interpolated1 - interpolated2) ** 2

    field_line_invariant_clipped = torch.where(unclipped_mask, field_line_invariant, field_line_invariant.mean(dim=1, keepdim=True))
    min_field_line_invariant, max_field_line_invariant = torch.aminmax(field_line_invariant_clipped, dim=1)
    field_line_invariant_range = max_field_line_invariant - min_field_line_invariant
    error_diff = torch.sqrt(torch.mean(interp_diff, dim=1) / 2) / field_line_invariant_range

    # infinity if 0 field_line_invariant range
    error_diff = torch.where(field_line_invariant_range > 0, error_diff, torch.inf)

    # peak field_line_invariant must be on top
    peak_field_line_invariant = field_line_invariant.gather(1, inflection_points.unsqueeze(1)).squeeze(1)
    average_field_line_invariant = field_line_invariant.quantile(0.85, dim=1)
    error_diff = torch.where(peak_field_line_invariant > average_field_line_invariant, error_diff, torch.inf)

    # require a minimum amount of each branch after trimming
    max_clip = max_clip if max_clip is not None else duration // 2
    error_diff = torch.where((((unclipped_mask & before).long().sum(dim=1) >= duration // 4)
                              & ((unclipped_mask & after).long().sum(dim=1) >= duration // 4)
                              & (unclipped_mask.long().sum(dim=1) >= duration - max_clip)
                              ), error_diff, torch.inf)

    return error_diff


def _calculate_residue_fit(potential_array,
                           transverse_field_line_invariant):
    min_field_line_invariant, max_field_line_invariant = torch.aminmax(transverse_field_line_invariant)
    field_line_invariant_range = max_field_line_invariant - min_field_line_invariant

    potential_array = potential_array.cpu().numpy()
    transverse_field_line_invariant = transverse_field_line_invariant.cpu().numpy()

    coeffs = np.polyfit(x=potential_array, y=transverse_field_line_invariant, deg=3)
    field_line_invariant_fit = np.poly1d(coeffs)(potential_array)
    rmse = np.sqrt(np.mean((transverse_field_line_invariant - field_line_invariant_fit) ** 2))
    error_fit = rmse / field_line_invariant_range

    return error_fit
