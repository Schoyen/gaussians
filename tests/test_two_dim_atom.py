import numpy as np
import scipy.linalg

from gaussians import G2D
import gaussians.two_dim_lib as tdl

from quantum_systems import BasisSet


def test_origin_invariance():
    center = (2 * (np.random.random() - 0.5), 2 * (np.random.random() - 0.5))
    a = 2

    gaussians_origin = [G2D((i, j), a) for i in range(3) for j in range(3)]
    gaussians = [G2D((i, j), a, A=center) for i in range(3) for j in range(3)]

    g2d_o_params = [g.get_params() for g in gaussians_origin]
    g2d_params = [g.get_params() for g in gaussians]

    bs_origin = BasisSet(len(gaussians_origin), dim=2)
    bs = BasisSet(len(gaussians), dim=2)

    bs_origin.h = tdl.construct_kinetic_operator_matrix_elements(
        g2d_o_params
    ) - tdl.construct_coulomb_attraction_operator_matrix_elements(
        (0, 0), g2d_o_params
    )
    bs.h = tdl.construct_kinetic_operator_matrix_elements(
        g2d_params
    ) - tdl.construct_coulomb_attraction_operator_matrix_elements(
        center, g2d_params
    )

    bs_origin.s = tdl.construct_overlap_matrix_elements(g2d_o_params)
    bs.s = tdl.construct_overlap_matrix_elements(g2d_params)

    eps_o, C_o = scipy.linalg.eigh(bs_origin.h, bs_origin.s)
    eps, C = scipy.linalg.eigh(bs.h, bs.s)

    np.testing.assert_allclose(eps_o, eps)
