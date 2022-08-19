import torch
from torchinterp1d import Interp1d
import numpy as np

interp = Interp1d()


def _calculate_residue_diff(inflection_points, potential, pressure):
    assert len(inflection_points.shape) == 1
    assert len(potential.shape) == 2
    assert len(pressure.shape) == 2

    if len(inflection_points) == 0:
        return torch.empty((0,), device=inflection_points.device, dtype=torch.float32)

    interp = Interp1d()

    min_pressure, max_pressure = torch.aminmax(pressure, dim=1)
    pressure_range = max_pressure - min_pressure

    potential = torch.where(potential.gather(1, inflection_points.unsqueeze(1)) < 0, -potential, potential)

    # second dim is potential1, pressure1, potential2, pressure2
    folded_data = torch.concat((potential.unsqueeze(1), pressure.unsqueeze(1)), dim=1).repeat(1, 2, 1)
    duration = folded_data.shape[2]

    all_indices = torch.arange(duration, device=inflection_points.device)
    all_indices = all_indices.unsqueeze(0).unsqueeze(1).expand(folded_data.shape[0], 2, -1)
    before = all_indices < inflection_points.unsqueeze(1).unsqueeze(2)
    after = all_indices > inflection_points.unsqueeze(1).unsqueeze(2)

    indices = torch.arange(len(folded_data), device=folded_data.device)
    inflection_point_values = folded_data[indices, :2, inflection_points]
    inflection_point_values = inflection_point_values.unsqueeze(2).expand(-1, -1, duration)

    # in first half, values after inflection point should be the inflection points repeated
    folded_data[:, :2, :] = torch.where(after, inflection_point_values, folded_data[:, :2, :])

    # in second half, values before inflection point should be the inflection points repeated
    folded_data[:, 2:, :] = torch.where(before, inflection_point_values, folded_data[:, 2:, :])

    sort1 = torch.argsort(folded_data[:, 0, :], dim=1)
    folded_data[:, 0, :] = folded_data[:, 0, :].gather(1, sort1)
    folded_data[:, 1, :] = folded_data[:, 1, :].gather(1, sort1)

    sort2 = torch.argsort(folded_data[:, 2, :], dim=1)
    folded_data[:, 2, :] = folded_data[:, 2, :].gather(1, sort2)
    folded_data[:, 3, :] = folded_data[:, 3, :].gather(1, sort2)

    interp_potential = torch.sort(potential, dim=1)[0]
    interpolated1 = interp.forward(x=folded_data[:, 0, :],
                                   y=folded_data[:, 1, :],
                                   xnew=interp_potential)
    interpolated2 = interp.forward(x=folded_data[:, 2, :],
                                   y=folded_data[:, 3, :],
                                   xnew=interp_potential)

    limit = torch.clamp(folded_data[:, 2, 0], min=0)
    limit_mask = interp_potential >= limit.unsqueeze(1)
    interp_diff = torch.where(limit_mask, (interpolated1 - interpolated2) ** 2, 0)

    # normalize by number of data points (subtract one to exclude inflection point)
    sample_counts = limit_mask.long().sum(dim=1) - 1
    interp_diff = torch.where(sample_counts.unsqueeze(1) > 1, interp_diff, torch.inf)

    error_diff = torch.sqrt(torch.sum(interp_diff, dim=1) / (sample_counts * 2)) / pressure_range
    return torch.where(pressure_range > 0, error_diff, torch.inf)


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
