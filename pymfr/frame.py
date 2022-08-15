import torch


def estimate_ht_frame(magnetic_field: torch.tensor,
                      electric_field: torch.tensor):
    """
    Estimate the DeHoffman-Teller (HT) frame based on magnetic field and electric field data such that
    the electric field in such a frame is zero.

    References:
    Paschmann, Gotz, and Bengt U. O. Sonnerup.
    “Proper Frame Determination and Walen Test.” ISSI Scientific Reports Series 8 (January 1, 2008): 65–74.

    :param magnetic_field: Tensor with shape ([B, ]N, 3).
    :param electric_field: Tensor with shape ([B, ]N, 3).
    If electric field measurements are unavailable, it can be estimates at -(v cross B).
    If available, electron velocity should be used to estimate the electric field.
    Otherwise, proton velocity can be used.
    :return Tensor with shape (B, 3)
    """

    assert magnetic_field.shape == electric_field.shape
    assert len(magnetic_field.shape) >= 2
    assert magnetic_field.shape[-1] == 3
    assert magnetic_field.device == electric_field.device

    Bx, By, Bz = magnetic_field[..., 0], magnetic_field[..., 1], magnetic_field[..., 2]

    coefficients = [[By ** 2 + Bz ** 2, -Bx * By, -Bx * Bz],
                    [-Bx * By, Bx ** 2 + Bz ** 2, -By * Bz],
                    [-Bx * Bz, -By * Bz, Bx ** 2 + By ** 2]]
    coefficient_matrix = torch.zeros((*magnetic_field.shape[:-2], 3, 3), device=magnetic_field.device)
    for i in range(3):
        for j in range(3):
            coefficient_matrix[..., i, j] = coefficients[i][j].mean(dim=-1)

    dependent_values = torch.cross(electric_field, magnetic_field, dim=-1).mean(dim=-2)

    fitting_result = torch.linalg.lstsq(coefficient_matrix, dependent_values)
    return fitting_result.solution


def estimate_ht2d_frame(magnetic_field: torch.tensor,
                        electric_field: torch.tensor,
                        axes: torch.tensor):
    """
    Find the frame where the electric field along the axis is zero.
    Assumes that the covariance matrix of the magnetic field
    is invertible and that there is no perfect DeHoffman-Teller frame.
    If this becomes an issue, a possible solution would be to
    use least square solution instead of exactly solving
    the linear system, but that needs to be investigated first.

    References:
    Paschmann, Gotz, and Bengt U. O. Sonnerup.
    “Proper Frame Determination and Walen Test.” ISSI Scientific Reports Series 8 (January 1, 2008): 65–74.

    :param magnetic_field: Tensor with shape ([B, ]N, 3)
    :param electric_field: Tensor with shape ([B, ]N, 3)
    :param axes: Tensor with shape ([B, [M, ]], 3)
    The covariance matrices will be calculated once for
    each batch and then the frame will be calculated for each axis
    if multiple axes are provided.
    :return: Tensor with shape ([B, [M, ]], 3)
    """

    assert magnetic_field.shape == electric_field.shape
    assert len(magnetic_field.shape) >= 2
    assert magnetic_field.shape[-1] == axes.shape[-1] == 3
    assert magnetic_field.device == electric_field.device == axes.device
    multiple_axes = len(axes.shape) > 2
    if multiple_axes:
        assert axes.shape[:-2] == magnetic_field.shape[:-2]
    else:
        assert axes.shape[:-1] == magnetic_field.shape[:-2]

    MBB = _covariance(magnetic_field, magnetic_field)
    MBE = _covariance(magnetic_field, electric_field)
    solutions = torch.linalg.solve(MBB, MBE)

    if multiple_axes:
        # add extra dimension after batch dimensions for axes
        solutions = solutions.unsqueeze(-3).expand(*((-1,) * (len(axes.shape) - 2)), axes.shape[-2], -1, -1)

    frames = torch.cross(axes, (solutions @ axes.unsqueeze(-1)).squeeze(-1), dim=-1)
    return frames


def _covariance(a, b):
    assert a.shape == b.shape
    assert len(a.shape) >= 2

    d_a = a - a.mean(dim=-2, keepdim=True)
    d_b = b - b.mean(dim=-2, keepdim=True)
    matrix = torch.zeros((*a.shape[:-2], a.shape[-1], a.shape[-1]), device=a.device)
    for i in range(3):
        for j in range(3):
            matrix[..., i, j] = (d_a[..., i] * d_b[..., j]).mean(dim=-1)
    return matrix


def _find_frames(magnetic_field, velocity, trial_axes, frame_type):
    n_batch = len(magnetic_field)
    batch_axes = torch.repeat_interleave(trial_axes, repeats=n_batch, dim=0)

    electric_field = -torch.cross(velocity, magnetic_field, dim=2)

    if frame_type == "mean_velocity":
        batch_frames = velocity.mean(dim=1).repeat(len(trial_axes), 1)
    elif frame_type == "vht_2d":
        axes = trial_axes.unsqueeze(0).expand(n_batch, -1, -1)
        batch_frames = estimate_ht2d_frame(magnetic_field, electric_field, axes)

        # output will be (batch, axis, 3), but we made each axis repeated "batch" times,
        # so the same axis should be repeated and the batch should change every entry,
        # so before flattening, need to transpose
        batch_frames = batch_frames.transpose(0, 1).reshape(-1, 3)
    elif frame_type == "vht":
        vht = estimate_ht_frame(magnetic_field, electric_field)
        batch_frames = vht.repeat(len(trial_axes), 1)
    else:
        raise Exception(f"Unknown frame type {frame_type}")

    return batch_frames, batch_axes
