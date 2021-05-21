import numpy as np

from gaussians import G1D, OD1D
from gaussians.one_dim.diff_mm_operator import construct_diff_mm_matrix_elements

import gaussians.one_dim_lib as odl

import helpers.utils
import helpers.multipole_moment


def test_overlap():
    gaussians = [G1D(0, 1, 0.5), G1D(0, 0.5, 0), G1D(1, 1, 0), G1D(2, 1, 0)]

    s = helpers.multipole_moment.construct_multipole_moment_matrix_elements(
        0, 0, gaussians
    )
    s_2 = helpers.multipole_moment.construct_overlap_matrix(gaussians)
    s_5 = helpers.multipole_moment.construct_overlap_matrix_elements(gaussians)
    s_6 = construct_diff_mm_matrix_elements(0, 0, 0, gaussians)

    x = np.linspace(-10, 10, 1001)

    s_3 = np.zeros_like(s_2)
    s_4 = np.zeros_like(s_2)

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            s_3[i, j] = G_i.norm * G_j.norm * np.trapz(G_i(x) * G_j(x), x=x)
            s_4[i, j] = (
                G_i.norm
                * G_j.norm
                * helpers.utils.overlap(
                    G_i.a, G_i.i, G_i.A, G_j.a, G_j.i, G_j.A
                )
            )

    np.testing.assert_allclose(s, s_2)
    np.testing.assert_allclose(s_3, s_4, atol=1e-12)
    np.testing.assert_allclose(s, s_4)
    np.testing.assert_allclose(s, s_3, atol=1e-12)
    np.testing.assert_allclose(s_2, s_2.T)
    np.testing.assert_allclose(s, s.T)
    np.testing.assert_allclose(s, s_5)
    np.testing.assert_allclose(s, s_6)

    np.testing.assert_allclose(
        s_5,
        odl.construct_overlap_matrix_elements(
            [g.get_params() for g in gaussians]
        ),
    )


def test_dipole_moment():
    gaussians = [G1D(0, 2, -4), G1D(0, 2, 4), G1D(1, 1, 0), G1D(2, 1, 0)]

    d = helpers.multipole_moment.construct_multipole_moment_matrix_elements(
        1, 1, gaussians
    )
    d_2 = helpers.multipole_moment.construct_dipole_moment_matrix(gaussians)
    d_3 = construct_diff_mm_matrix_elements(1, 0, 1, gaussians)

    np.testing.assert_allclose(d, d_2)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(d, d_3)

    np.testing.assert_allclose(
        d,
        odl.construct_multipole_moment_matrix_elements(
            1, 1, tuple(g.get_params() for g in gaussians)
        ),
    )
