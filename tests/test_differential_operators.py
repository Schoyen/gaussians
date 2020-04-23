import numpy as np

from gaussians import G1D, OD1D
from gaussians.one_dim.differential_operator import (
    construct_differential_matrix_elements,
)
from gaussians.one_dim.multipole_moment import S


# def kinetic(G_i, G_j):
#     b = G_j.a
#     j = G_j.i
#     B = G_j.A
#
#     return (
#         -2 * b ** 2 * S(0, 0, G_i, G1D(j + 2, b, B))
#         + b * (2 * j + 1) * S(0, 0, G_i, G_j)
#         - (
#             (0.5 * j * (j - 1) * S(0, 0, G_i, G1D(j - 2, b, B)))
#             if j >= 2
#             else 0
#         )
#     )


def kinetic(G_i, G_j):
    b = G_j.a
    j = G_j.i
    B = G_j.A

    kin = 4 * b ** 2 * S(0, 0, G_i, G1D(j + 2, b, B))
    kin -= 2 * b * (2 * j + 1) * S(0, 0, G_i, G_j)
    kin += j * (j - 1) * S(0, 0, G_i, G1D(j - 2, b, B)) if j > 1 else 0

    return -0.5 * kin


# def kinetic(G_i, G_j):
#     a = G_i.a
#     i = G_i.i
#     A = G_i.A
#
#     kin = 4 * a ** 2 * S(0, 0, G1D(i + 2, a, A), G_j)
#     kin -= 2 * a * (2 * i + 1) * S(0, 0, G_i, G_j)
#     kin += i * (i - 1) * S(0, 0, G1D(i - 2, a, A), G_j) if i > 1 else 0
#
#     return -0.5 * kin


def construct_kinetic_matrix_elements(gaussians):
    l = len(gaussians)
    t = np.zeros((l, l))

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            t[i, j] = G_i.norm * G_j.norm * kinetic(G_i, G_j)

    return t


def test_kinetic_elements():
    gaussians = [G1D(0, 2, -4), G1D(0, 2, 4)]

    t = construct_kinetic_matrix_elements(gaussians)

    assert abs(t[0, 0] - t[1, 1]) < 1e-12
    assert abs(t[0, 1] - t[1, 0]) < 1e-12
    assert abs(t[0, 1]) < 1e-12

    t_2 = -0.5 * construct_differential_matrix_elements(2, gaussians)
    print(t_2)
    print(t)
    np.testing.assert_allclose(t, t_2)
