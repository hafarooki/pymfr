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
    assert magnetic_field.dtype == electric_field.dtype
    assert len(magnetic_field.shape) >= 2
    assert magnetic_field.shape[-1] == 3
    assert magnetic_field.device == electric_field.device

    Bx, By, Bz = magnetic_field[..., 0], magnetic_field[..., 1], magnetic_field[..., 2]

    coefficients = [[By ** 2 + Bz ** 2, -Bx * By, -Bx * Bz],
                    [-Bx * By, Bx ** 2 + Bz ** 2, -By * Bz],
                    [-Bx * Bz, -By * Bz, Bx ** 2 + By ** 2]]
    coefficient_matrix = torch.zeros((*magnetic_field.shape[:-2], 3, 3), device=magnetic_field.device,
                                     dtype=magnetic_field.dtype)
    for i in range(3):
        for j in range(3):
            coefficient_matrix[..., i, j] = coefficients[i][j].mean(dim=-1)

    dependent_values = torch.cross(electric_field, magnetic_field, dim=-1).mean(dim=-2)

    # I am not sure why, but this seems to give better results than torch.linalg.lstsq on GPU
    # the pinv method is closer to the result form torch.linalg.lstsq on CPU and np.linalg.lstsq
    # furthermore, torch.linalg.lstsq sometimes give LinAlgError in batch mode the input matrix is not full rank,
    # even though it works fine if applied individually
    fitting_result = (torch.linalg.pinv(coefficient_matrix, hermitian=True) @ dependent_values.unsqueeze(2)).squeeze(2)
    return fitting_result


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
    assert magnetic_field.dtype == electric_field.dtype == axes.dtype
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
    assert a.dtype == b.dtype
    assert len(a.shape) >= 2

    d_a = a - a.mean(dim=-2, keepdim=True)
    d_b = b - b.mean(dim=-2, keepdim=True)
    matrix = torch.zeros((*a.shape[:-2], a.shape[-1], a.shape[-1]), device=a.device, dtype=a.dtype)
    for i in range(3):
        for j in range(3):
            matrix[..., i, j] = (d_a[..., i] * d_b[..., j]).mean(dim=-1)
    return matrix