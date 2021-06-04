import numpy as np

from gaussians import G1D, OD1D

from gaussians.one_dim.diff_mm_operator import (
    construct_diff_mm_matrix_elements,
    construct_kinetic_matrix_elements,
)

import gaussians.one_dim_lib as odl
import helpers.differential_operator


def test_momentum_elements():
    gaussians = [
        G1D(i, np.random.random() + 0.1, 2 * (np.random.random() - 0.5))
        for i in range(4)
    ]

    p = helpers.differential_operator.construct_differential_matrix_elements(
        1, gaussians
    )
    p_2 = construct_diff_mm_matrix_elements(0, 1, 0, gaussians)
    p_r = odl.construct_diff_mm_matrix_elements(
        0, 1, 0, [g.get_params() for g in gaussians]
    )

    np.testing.assert_allclose(p, p_2, atol=1e-12)
    np.testing.assert_allclose(p_2, p_r, atol=1e-12)

    for i in range(len(gaussians)):
        for j in range(len(gaussians)):
            if i == j:
                continue

            assert abs(p_2[i, j] + p_2[j, i]) < 1e-12


def test_kinetic_elements():
    gaussians = [G1D(0, 1, 0.5), G1D(0, 0.5, 0), G1D(1, 1, 0), G1D(2, 1, 0)]

    t = helpers.differential_operator.construct_kinetic_matrix(gaussians)
    t_2 = (
        -0.5
        * helpers.differential_operator.construct_differential_matrix_elements(
            2, gaussians
        )
    )
    t_3 = helpers.differential_operator.construct_kinetic_matrix_elements(
        gaussians
    )
    t_4 = -0.5 * construct_diff_mm_matrix_elements(0, 2, 0, gaussians)
    t_5 = construct_kinetic_matrix_elements(gaussians)

    np.testing.assert_allclose(t, t_2, atol=1e-12)
    np.testing.assert_allclose(t, t.T, atol=1e-12)
    np.testing.assert_allclose(t_2, t_3)
    np.testing.assert_allclose(t, t_4, atol=1e-12)
    np.testing.assert_allclose(t_4, t_5)

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
