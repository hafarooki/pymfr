import torch
from torchinterp1d import Interp1d
import numpy as np


def _calculate_residue_diff(inflection_points, potential, pressure, max_clip=None):
    assert len(inflection_points.shape) == 1
    assert len(potential.shape) == 2
    assert len(pressure.shape) == 2

    if len(inflection_points) == 0:
        return torch.empty((0,), device=inflection_points.device, dtype=torch.float32)

    # force to be positive at peak
    potential = torch.where(potential.gather(1, inflection_points.unsqueeze(1)) < 0, -potential, potential)

    # second dim is potential1, pressure1, potential2, pressure2
    folded_data = torch.concat((potential.unsqueeze(1), pressure.unsqueeze(1)), dim=1).repeat(1, 2, 1)
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

    pressure_clipped = torch.where(unclipped_mask, pressure, pressure.mean(dim=1, keepdim=True))
    min_pressure, max_pressure = torch.aminmax(pressure_clipped, dim=1)
    pressure_range = max_pressure - min_pressure
    error_diff = torch.sqrt(torch.mean(interp_diff, dim=1) / 2) / pressure_range

    # infinity if 0 pressure range
    error_diff = torch.where(pressure_range > 0, error_diff, torch.inf)

    # peak pressure must be on top
    peak_pressure = pressure.gather(1, inflection_points.unsqueeze(1)).squeeze(1)
    average_pressure = pressure.quantile(0.85, dim=1)
    error_diff = torch.where(peak_pressure > average_pressure, error_diff, torch.inf)

    return error_diff


def _calculate_residue_fit(potential_array,
                           transverse_pressure):
    min_pressure, max_pressure = torch.aminmax(transverse_pressure)
    pressure_range = max_pressure - min_pressure

    potential_array = potential_array.cpu().numpy()
    transverse_pressure = transverse_pressure.cpu().numpy()

    coeffs = np.polyfit(x=potential_array, y=transverse_pressure, deg=3)
    pressure_fit = np.poly1d(coeffs)(potential_array)
    rmse = np.sqrt(np.mean((transverse_pressure - pressure_fit) ** 2))
    error_fit = rmse / pressure_range

    return error_fit
