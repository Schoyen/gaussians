import numpy as np

from gaussians import G1D, OD1D
from gaussians.one_dim.differential_operator import (
    construct_differential_matrix_elements,
    construct_kinetic_matrix_elements,
)
from gaussians.one_dim.multipole_moment import S
from gaussians.one_dim.diff_mm_operator import construct_diff_mm_matrix_elements

import gaussians.one_dim_lib as odl


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


def test_kinetic_elements():
    gaussians = [G1D(0, 1, 0.5), G1D(0, 0.5, 0), G1D(1, 1, 0), G1D(2, 1, 0)]

    t = construct_kinetic_matrix(gaussians)
    t_2 = -0.5 * construct_differential_matrix_elements(2, gaussians)
    t_3 = construct_kinetic_matrix_elements(gaussians)
    t_4 = -0.5 * construct_diff_mm_matrix_elements(0, 2, 0, gaussians)

    np.testing.assert_allclose(t, t_2, atol=1e-12)
    np.testing.assert_allclose(t, t.T, atol=1e-12)
    np.testing.assert_allclose(t_2, t_3)
    np.testing.assert_allclose(t, t_4, atol=1e-12)

    np.testing.assert_allclose(
        t,
        odl.construct_kinetic_operator_matrix_elements(
            [g.get_params() for g in gaussians]
        ),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        t,
        -0.5
        * odl.construct_differential_operator_matrix_elements(
            2, [g.get_params() for g in gaussians]
        ),
        atol=1e-12,
    )
