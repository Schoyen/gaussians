import numpy as np

from .g1d import G1D


def kinetic(b, j, s_left, s_center, s_right):
    return (
        -2 * b ** 2 * s_right
        + b * (2 * j + 1) * s_center
        - 0.5 * j * (j - 1) * s_left
    )


def construct_kinetic_matrix(gaussians, s):
    r"""
    >>> from gaussians.one_dim.g1d import G1D
    >>> from gaussians.one_dim.overlap import construct_overlap_matrix
    >>> gaussians = [G1D(0, 2, -4), G1D(0, 2, 4)]
    >>> s = construct_overlap_matrix(gaussians)
    >>> t = construct_kinetic_matrix(gaussians, s)
    >>> t.shape
    (2, 2)
    >>> abs(t[0, 0] - t[1, 1]) < 1e-12
    True
    >>> abs(t[0, 1] - t[1, 0]) < 1e-12
    True
    >>> abs(t[0, 1]) < 1e-12
    True
    """

    l = len(gaussians)
    t = np.zeros((l, l))

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            b = G_j.a
            j = G_j.i

            s_left = 0 if j < 2 else s[i, j - 2]
            s_right = 0 if j > len(gaussians) - 3 else s[i, j + 2]

            t[i, j] = (
                G_i.norm * G_j.norm * kinetic(b, j, s_left, s[i, j], s_right)
            )

    return t
