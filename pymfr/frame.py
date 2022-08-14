import torch


def _find_frames(magnetic_field, velocity, trial_axes, frame_type):
    batch_axes = torch.repeat_interleave(trial_axes, repeats=len(magnetic_field), dim=0)

    electric_field = -torch.cross(velocity, magnetic_field, dim=2)

    if frame_type == "mean_velocity":
        batch_frames = velocity.mean(dim=1).repeat(len(trial_axes), 1)
    elif frame_type == "vht_2d":
        MBB = _covariance(magnetic_field, magnetic_field)
        MBE = _covariance(magnetic_field, electric_field)
        solutions = torch.linalg.solve(MBB, MBE)

        solutions = solutions.repeat(len(trial_axes), 1, 1)
        batch_frames = torch.cross(batch_axes, (solutions @ batch_axes.unsqueeze(2)).squeeze(2))

    elif frame_type == "vht":
        Bx = magnetic_field[:, :, 0]
        By = magnetic_field[:, :, 1]
        Bz = magnetic_field[:, :, 2]

        coefficients = [[(By ** 2 + Bz ** 2).mean(dim=1), (-Bx * By).mean(dim=1), (-Bx * Bz).mean(dim=1)],
                        [(-Bx * By).mean(dim=1), (Bx ** 2 + Bz ** 2).mean(dim=1), (-By * Bz).mean(dim=1)],
                        [(-Bx * Bz).mean(dim=1), (-By * Bz).mean(dim=1), (Bx ** 2 + By ** 2).mean(dim=1)]]
        coefficient_matrix = torch.stack(
            [torch.stack(x, dim=1) for x in coefficients], dim=2)

        dependent_values = torch.cross(electric_field, magnetic_field, dim=2).mean(dim=1)

        fitting_result = torch.linalg.lstsq(coefficient_matrix, dependent_values)
        vht = fitting_result.solution
        batch_frames = vht.repeat(len(trial_axes), 1)
    else:
        raise Exception(f"Unknown frame type {frame_type}")

    return batch_frames, batch_axes


def _covariance(x, y):
    dx = x - x.mean(dim=1, keepdim=True)
    dy = y - y.mean(dim=1, keepdim=True)
    coefficients = [[(dx[:, :, i] * dy[:, :, j]).mean(dim=1) for j in range(3)]
                    for i in range(3)]
    rows = [torch.stack(columns, dim=1) for columns in coefficients]
    stack = torch.stack(rows, dim=1)
    return stack
