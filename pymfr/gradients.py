from array_api_compat import array_namespace


def minrmsdd(gradients):
    """
    Given `gradients` (..., N, 3, 3), defined so gradient_ij = dBi/dxj, return the
    min-root-mean-square directional derivative unit vector (..., 3).

    Note: Array API compatible.
    """
    xp = array_namespace(gradients)

    if len(gradients.shape) < 3:
        raise ValueError("gradients must have at least 3 dimensions")

    # G^T G, so that n^T (G^T G) n = |G n|^2  (..., N, 3, 3)
    symmetric_tensors = xp.matrix_transpose(gradients) @ gradients

    eigenvalues, eigenvectors = xp.linalg.eigh(xp.mean(symmetric_tensors, axis=-3))

    # the Array API leaves eigh's eigenvalue order unspecified, so pick the
    # min eigenvector column explicitly  (..., 1, 1)
    argmin = xp.argmin(eigenvalues, axis=-1, keepdims=True)[..., None]

    return xp.take_along_axis(eigenvectors, argmin, axis=-1)[..., 0]
