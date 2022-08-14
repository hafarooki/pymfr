import torch
from torchinterp1d import Interp1d
import numpy as np

interp = Interp1d()


def _calculate_residue_diff(inflection_point, potential_array,
                            transverse_pressure):
    min_pressure, max_pressure = torch.aminmax(transverse_pressure)
    pressure_range = max_pressure - min_pressure

    if pressure_range == 0:
        return 1

    potential1 = potential_array[:inflection_point + 1]
    pressure1 = transverse_pressure[:inflection_point + 1]
    potential2 = torch.flip(potential_array[inflection_point:], dims=(0,))
    pressure2 = torch.flip(transverse_pressure[inflection_point:], dims=(0,))
    interp_potential = torch.sort(potential_array)[0]
    sort1 = torch.argsort(potential1)
    sort2 = torch.argsort(potential2)
    interpolated1 = interp.forward(x=potential1[sort1], y=pressure1[sort1], xnew=interp_potential)
    interpolated2 = interp.forward(x=potential2[sort2], y=pressure2[sort2], xnew=interp_potential)
    interp_diff = interpolated1 - interpolated2
    error_diff = torch.sqrt(torch.mean(interp_diff ** 2) / 2) / pressure_range

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
