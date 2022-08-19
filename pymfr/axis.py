import numpy as np
import scipy.constants
import torch
import torch.nn.functional as F

from pymfr.residue import _calculate_residue_diff


def minimize_rdiff(magnetic_field, gas_pressure, frame_velocity, iterations=3):
    """
    WIP API for finding best axis using Rdiff as a criteria
    :param magnetic_field:
    :param frame_velocity:
    :param iterations:
    :return:
    """

    best_axis = None
    best_rdiff = None

    batch_axes = _get_trial_axes(range(0, 90, 1), range(0, 360, 2)).to(magnetic_field.device)
    for i in range(iterations):
        batch_field = magnetic_field.unsqueeze(0).expand(len(batch_axes), -1, -1)
        batch_frames = frame_velocity.unsqueeze(0).expand(len(batch_axes), -1)
        batch_gas_pressure = gas_pressure.unsqueeze(0).expand(len(batch_axes), -1)

        rotated_field = _rotate_field(batch_axes, batch_field, batch_frames)
        transverse_pressure = batch_gas_pressure + (rotated_field[:, :, 2] * 1e-9) ** 2 / (2 * scipy.constants.mu_0) * 1e9
        potential = torch.zeros((len(rotated_field)), rotated_field.shape[1], device=rotated_field.device)
        potential[:, 1:] = torch.cumulative_trapezoid(rotated_field[:, :, 1])

        inflection_points = potential.abs().argmax(dim=1)

        peaks = transverse_pressure[torch.arange(transverse_pressure.shape[0], device=transverse_pressure.device),
                                    inflection_points]
        min_pressure, max_pressure = torch.aminmax(transverse_pressure, dim=1)
        thresholds = torch.quantile(transverse_pressure, 0.85, dim=1, interpolation="lower")

        mask = (inflection_points > 0) & \
               (inflection_points < (potential.shape[-1] - 1)) & \
               (peaks > thresholds) & \
               ((potential.abs()[:, -1] / potential.abs().amax(dim=1)) < 0.1) & \
               ((max_pressure - min_pressure) > 0)

        rdiff = _calculate_residue_diff(inflection_points, potential, transverse_pressure)
        rdiff = torch.where(mask, rdiff, torch.inf)

        argmin = torch.argmin(rdiff)
        if best_rdiff is None or rdiff[argmin] < best_rdiff:
            if rdiff[argmin] < torch.inf:
                best_axis = batch_axes[argmin]
                best_rdiff = rdiff[argmin]

        if best_axis is None:
            break

        x, y, z = tuple(best_axis.cpu().numpy())
        altitude = np.rad2deg(np.arctan2(np.sqrt(x ** 2 + y ** 2), z))
        azimuth = np.rad2deg(np.arctan2(y, x))
        batch_axes = _get_trial_axes(np.linspace(altitude - 30 / (i + 1), altitude + 30 / (i + 1), 50),
                                     np.linspace(azimuth - 60 / (i + 1), azimuth + 60 / (i + 1), 50))
        batch_axes = batch_axes.to(magnetic_field.device)

    return best_axis, best_rdiff


def _get_trial_axes(altitude_range, azimuth_range):
    trial_axes = [torch.tensor([np.sin(np.deg2rad(altitude)) * np.cos(np.deg2rad(azimuth)),
                                np.sin(np.deg2rad(altitude)) * np.sin(np.deg2rad(azimuth)),
                                np.cos(np.deg2rad(altitude))], dtype=torch.float32)
                  for altitude in altitude_range
                  for azimuth in azimuth_range]
    trial_axes.append(torch.tensor([0, 0, 1], dtype=torch.float32))
    trial_axes = torch.row_stack(trial_axes)
    return trial_axes


def _rotate_field(batch_axes, batch_field, batch_frames):
    z_unit = batch_axes
    x_unit = F.normalize(-(batch_frames - (batch_frames * z_unit).sum(dim=1).unsqueeze(1) * z_unit))
    y_unit = torch.cross(z_unit, x_unit)
    rotation_matrix = torch.stack([x_unit, y_unit, z_unit], dim=2)
    rotation_matrix = rotation_matrix.transpose(1, 2)  # transpose gives inverse of rotation matrix
    rotated_field = (rotation_matrix @ batch_field.transpose(1, 2)).transpose(1, 2)
    return rotated_field
