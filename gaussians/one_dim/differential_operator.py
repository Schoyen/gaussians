import numpy as np

from .g1d import G1D
from .multipole_moment import S


def construct_differential_matrix_elements(
    e: int, gaussians: list
) -> np.ndarray:
    l = len(gaussians)
    d_e = np.zeros((l, l))

    for i in range(l):
        G_i = gaussians[i]
        d_e[i, i] = G_i.norm ** 2 * D(e, G_i, G_i)

        for j in range(i + 1, l):
            G_j = gaussians[j]
            val = G_i.norm * G_j.norm * D(e, G_i, G_j)

            d_e[i, j] = val
            d_e[j, i] = val

    return d_e


def D(e: int, G_i: G1D, G_j: G1D) -> float:
    assert e >= 0

    if e == 0:
        return S(0, 0, G_i, G_j)

    i = G_i.i
    a = G_i.a
    A = G_i.A

    forward = 2 * a * D(e - 1, G1D(i + 1, a, A), G_j)
    backward = 0 if i < 1 else (-i * D(e - 1, G1D(i - 1, a, A), G_j))

    return forward + backward
