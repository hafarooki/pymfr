import numpy as np
import scipy.constants
import torch
import torch.nn.functional as F

from pymfr.residue import _calculate_residue_diff


def minimize_rdiff(magnetic_field, frame_velocity, trial_axes, max_clip=None):
    """
    WIP API for finding best axis using Rdiff as a criteria
    """

    rdiff = calculate_residue_map(magnetic_field, frame_velocity, trial_axes, max_clip=max_clip)
    rdiff, argmin = rdiff.min(dim=-1)
    index = argmin.unsqueeze(-1).unsqueeze(-1).expand(*[-1] * len(magnetic_field.shape[:-2]), -1, 3)
    trial_axes = trial_axes.gather(-2, index).squeeze(-2)
    return trial_axes, rdiff


def calculate_residue_map(magnetic_field, frame_velocity, trial_axes, max_clip=None):
    """
    WIP API
    """

    batch_size = magnetic_field.shape[:-2]
    duration = magnetic_field.shape[-2]
    n_trial_axes = trial_axes.shape[-2]
    assert magnetic_field.device == frame_velocity.device == trial_axes.device
    assert magnetic_field.shape == (*batch_size, duration, 3)
    assert frame_velocity.shape == (*batch_size, 3) or frame_velocity.shape == (*batch_size, n_trial_axes, 3)
    assert trial_axes.shape == (*batch_size, n_trial_axes, 3)
    device = magnetic_field.device
    trial_field = magnetic_field.unsqueeze(len(batch_size)).expand(*([-1] * len(batch_size)), n_trial_axes, -1, -1)
    if frame_velocity.shape == (*batch_size, 3):
        trial_frame = frame_velocity.unsqueeze(len(batch_size)).expand(*([-1] * len(batch_size)), n_trial_axes, -1)
    else:
        trial_frame = frame_velocity
    
    rotated_field = _rotate_field(trial_axes.reshape(-1, 3),
                                  trial_field.reshape(-1, duration, 3),
                                  trial_frame.reshape(-1, 3)).reshape(*batch_size, n_trial_axes, duration, 3)
    potential = torch.zeros(rotated_field.shape[:-1], device=device)
    potential[..., 1:] = torch.cumulative_trapezoid(rotated_field[..., 1])
    field_line_invariant = rotated_field[..., 2]
    inflection_points = potential[..., 1:-1].abs().argmax(dim=-1) + 1
    rdiff = _calculate_residue_diff(inflection_points.reshape(-1),
                                    potential.reshape(-1, duration),
                                    field_line_invariant.reshape(-1, duration),
                                    max_clip).reshape(*batch_size, n_trial_axes)
    return rdiff


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
    projected = (batch_frames * z_unit).sum(dim=1).unsqueeze(1) * z_unit
    x_unit = F.normalize(-(batch_frames - projected))
    y_unit = torch.cross(z_unit, x_unit)
    rotation_matrix = torch.stack([x_unit, y_unit, z_unit], dim=2)
    rotation_matrix = rotation_matrix.transpose(1, 2)  # transpose gives inverse of rotation matrix
    rotated_field = (rotation_matrix @ batch_field.transpose(1, 2)).transpose(1, 2)
    return rotated_field
