import numpy as np
import scipy.linalg

from quantum_systems import TwoDimensionalHarmonicOscillator

from gaussians import G2D
from gaussians.two_dim import (
    construct_coulomb_matrix_elements,
    construct_kinetic_matrix_elements,
    construct_overlap_matrix_elements,
    construct_multipole_moment_matrix_elements,
)


def test_construction():
    gaussians = [G2D((0, 0), 1), G2D((1, 0), 1), G2D((0, 1), 1), G2D((1, 1), 1)]

    u = construct_coulomb_matrix_elements(gaussians)

    np.testing.assert_allclose(u, u.transpose(2, 3, 0, 1))


def test_two_dim_ho():
    omega = 1
    l = 2

    gaussians = [G2D((i, j), omega / 2) for i in range(l) for j in range(l)]

    tdho = TwoDimensionalHarmonicOscillator(16, 5, 201, omega=omega)

    t = construct_kinetic_matrix_elements(gaussians)
    v = (
        0.5
        * omega ** 2
        * (
            construct_multipole_moment_matrix_elements(
                [2, 0], [0, 0], gaussians
            )
            + construct_multipole_moment_matrix_elements(
                [0, 2], [0, 0], gaussians
            )
        )
    )
    h = t + v
    s = construct_overlap_matrix_elements(gaussians)
    u = construct_coulomb_matrix_elements(gaussians)

    epsilon_2, C = scipy.linalg.eigh(h, s)

    n = 4

    X = tdho.R * np.cos(tdho.T)
    Y = tdho.R * np.sin(tdho.T)

    epsilon = np.diag(tdho.h)

    np.testing.assert_allclose(epsilon[:n], epsilon_2[:n], atol=1e-12)
    h_trans = tdho.transform_one_body_elements(h, C, np)
    np.testing.assert_allclose(tdho.h[:n, :n], h_trans[:n, :n], atol=1e-12)

    u_trans = tdho.transform_two_body_elements(u, C, np)

    np.testing.assert_allclose(tdho.u[:n, :n, :n, :n], u_trans[:n, :n, :n, :n])
