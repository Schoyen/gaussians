import numpy as np

from .multipole_moment import S


def construct_overlap_matrix(gaussians):
    r"""

    >>> from gaussians.one_dim.g1d import G1D
    >>> s = construct_overlap_matrix([G1D(0, 2, -4), G1D(0, 2, 4)])
    >>> s.shape
    (2, 2)
    >>> abs(s[0, 0] - s[1, 1]) < 1e-12
    True
    >>> abs(s[0, 1] - s[1, 0]) < 1e-12
    True
    >>> abs(s[0, 1]) < 1e-12
    True
    """

    l = len(gaussians)
    s = np.zeros((l, l))

    for i in range(l):
        G_i = gaussians[i]
        for j in range(i + 1, l):
            G_j = gaussians[j]

            val = S(0, 0, G_i, G_j)
            s[i, j] = val
            s[j, i] = val

    return s
