import numpy as np

from .g1d import G1D
from .overlap import overlap


def kinetic(G_i, G_j):

    return (
        -2 * G_j.a ** 2 * overlap(G_i, G1D(G_j.i + 2, G_j.a, G_j.A))
        + G_j.a * (2 * G_j.i + 1) * overlap(G_i, G_j)
        - 0.5 * G_j.i * (G_j.i - 1) * overlap(G_i, G1D(G_j.i - 2, G_j.a, G_j.A))
    )


def construct_kinetic_matrix(gaussians):
    r"""
    >>> from gaussians.one_dim.g1d import G1D
    >>> t = construct_kinetic_matrix([G1D(0, 2, -4), G1D(0, 2, 4)])
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
            t[i, j] = kinetic(G_i, G_j)

    return t
