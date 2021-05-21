import numpy as np

from gaussians import G1D
from .multipole_moment import S


def kinetic(G_i, G_j):
    b = G_j.a
    j = G_j.i
    B = G_j.A

    kin = 4 * b ** 2 * S(0, 0, G_i, G1D(j + 2, b, B))
    kin -= 2 * b * (2 * j + 1) * S(0, 0, G_i, G_j)
    kin += j * (j - 1) * S(0, 0, G_i, G1D(j - 2, b, B)) if j > 1 else 0

    return -0.5 * kin


def construct_kinetic_matrix(gaussians):
    l = len(gaussians)
    t = np.zeros((l, l))

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            t[i, j] = G_i.norm * G_j.norm * kinetic(G_i, G_j)

    return t


def construct_kinetic_matrix_elements(gaussians: list):
    return -0.5 * construct_differential_matrix_elements(2, gaussians)


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
