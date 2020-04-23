import numpy as np

from .g1d import G1D
from .multipole_moment import S


def construct_differential_matrix_elements(
    e: int, gaussians: list
) -> np.ndarray:
    l = len(gaussians)
    d_e = np.zeros((l, l))

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            d_e[i, j] = G_i.norm * G_j.norm * D_left(e, G_i, G_j)
            print("Hello")
            print(i, j)
            np.testing.assert_allclose(
                d_e[i, j], (-1) ** e * D_right(e, G_i, G_j)
            )

    return d_e


def D_left(e: int, G_i: G1D, G_j: G1D) -> float:
    assert e >= 0

    if e == 0:
        return S(0, 0, G_i, G_j)

    i = G_i.i
    a = G_i.a
    A = G_i.A

    forward = 2 * a * D_left(e - 1, G1D(i + 1, a, A), G_j)
    backward = 0 if i < 1 else (-i * D_left(e - 1, G1D(i - 1, a, A), G_j))

    return forward + backward


def D_right(e: int, G_i: G1D, G_j: G1D) -> float:
    assert e >= 0

    if e == 0:
        return S(0, 0, G_i, G_j)

    j = G_j.i
    b = G_j.a
    B = G_j.A

    forward = 2 * b * D_right(e - 1, G_i, G1D(j + 1, b, B))
    backward = 0 if j < 1 else (-j * D_right(e - 1, G_i, G1D(j - 1, b, B)))

    return forward + backward
