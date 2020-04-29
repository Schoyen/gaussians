import numpy as np
import scipy.linalg

# from helpers.utils import two_dim_grid_solver
from quantum_systems import TwoDimensionalHarmonicOscillator

from gaussians import G2D
from gaussians.two_dim import (
    construct_kinetic_matrix_elements,
    construct_overlap_matrix_elements,
    construct_multipole_moment_matrix_elements,
)


def test_two_dim_ho():
    omega = 1
    l = 10

    gaussians = [
        G2D((i, j), omega / np.sqrt(2)) for i in range(l) for j in range(l)
    ]

    # potential = lambda x, y, omega=omega: 0.5 * omega ** 2 * (x ** 2 + y ** 2)
    # epsilon, phi, X, Y = two_dim_grid_solver(potential)
    tdho = TwoDimensionalHarmonicOscillator(l, 5, 201, omega=omega)

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

    epsilon_2, C = scipy.linalg.eigh(h, s)

    n = 10

    epsilon = np.diag(tdho.h)

    np.testing.assert_allclose(epsilon[:n], epsilon_2[:n], atol=1e-3)
    h = np.einsum("ip, ij, jq -> pq", C.conj(), h, C, optimize=True)
    np.testing.assert_allclose(tdho.h[:n, :n], h[:n, :n], atol=1e-3)

    # Testing the single-paricle states will be hard as the grid solver
    # does not prefer a specific direction for the basis functions thus making
    # the states different from the analytical solution.
    # For the harmonic oscillator basis it gets even worse as the
    # single-particle states are rotationally invariant and therefore look like
    # circular hats.

    # import matplotlib.pyplot as plt

    # X = tdho.R * np.cos(tdho.T)
    # Y = tdho.R * np.sin(tdho.T)

    # for i in range(n):
    #     plt.figure()
    #     plt.title(f"Grid: {i}")
    #     plt.contourf(X, Y, np.abs(tdho.spf[i]) ** 2)

    # spf = np.tensordot(C, np.asarray([gauss(X, Y, with_norm=True) for gauss in gaussians]), axes=((0), (0)))

    # for i in range(n):
    #     plt.figure()
    #     plt.title(f"Analytic {i}")
    #     plt.contourf(X, Y, np.abs(spf[i]) ** 2)

    # plt.show()
