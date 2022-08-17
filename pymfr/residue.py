import torch
from torchinterp1d import Interp1d
import numpy as np

interp = Interp1d()


def _calculate_residue_diff(inflection_points, potential, pressure):
    interp = Interp1d()

    min_pressure, max_pressure = torch.aminmax(pressure, dim=1)
    pressure_range = max_pressure - min_pressure

    combined_data = torch.concat((potential.unsqueeze(1), pressure.unsqueeze(1)), dim=1)

    # second dim is potential1, pressure1, potential2, pressure2
    folded_data = combined_data[:, :, :].repeat(1, 2, 1)

    points = inflection_points

    all_indices = torch.arange(combined_data.shape[2], device=points.device)
    all_indices = all_indices.unsqueeze(0).unsqueeze(1).expand(combined_data.shape[0], 2, -1)
    before = all_indices < points.unsqueeze(1).unsqueeze(2)
    after = all_indices > points.unsqueeze(1).unsqueeze(2)

    inflection_point_values = combined_data[torch.arange(len(combined_data), device=combined_data.device), :, points]
    inflection_point_values = inflection_point_values.unsqueeze(2).expand(-1, -1, combined_data.shape[2])

    # in first half, values after inflection point should be the inflection points repeated
    folded_data[:, :2, :][before] = inflection_point_values[before]

    # in second half, values before inflection point should be the inflection points repeated
    folded_data[:, 2:, :][after] = inflection_point_values[after]

    interp_potential = torch.sort(potential, dim=1)[0]
    interpolated1 = interp.forward(x=folded_data[:, 0, :],
                                   y=folded_data[:, 1, :],
                                   xnew=interp_potential)
    interpolated2 = interp.forward(x=folded_data[:, 2, :],
                                   y=folded_data[:, 3, :],
                                   xnew=interp_potential)

    interp_diff = interpolated1 - interpolated2

    return torch.sqrt(torch.mean(interp_diff ** 2, dim=1) / 2) / pressure_range


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
