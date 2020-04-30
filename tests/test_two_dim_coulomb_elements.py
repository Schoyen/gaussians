import numpy as np
import scipy.linalg

from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    BasisSet,
    GeneralOrbitalSystem,
)
from hartree_fock import GHF

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

    tdho = GeneralOrbitalSystem(
        2, TwoDimensionalHarmonicOscillator(l ** 2, 5, 201, omega=omega)
    )

    ghf_ho = GHF(tdho, verbose=True)
    ghf_ho.compute_ground_state()

    X = tdho._basis_set.R * np.cos(tdho._basis_set.T)
    Y = tdho._basis_set.R * np.sin(tdho._basis_set.T)

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

    gos = BasisSet(len(gaussians), dim=2)
    gos.h = h
    gos.s = s
    gos.u = u

    spf = np.asarray([gauss(X, Y, with_norm=True) for gauss in gaussians])
    gos.spf = spf

    gos = GeneralOrbitalSystem(2, gos)

    ghf_gauss = GHF(gos, verbose=True)
    ghf_gauss.compute_ground_state()

    assert abs(ghf_ho.compute_energy() - ghf_gauss.compute_energy()) < 1e-12

    # import matplotlib.pyplot as plt

    # for i in range(0, tdho.l, 2):
    #     plt.figure()
    #     plt.title(f"Grid: {i}")
    #     plt.contourf(X, Y, np.abs(tdho.spf[i]) ** 2)

    # ghf_ho.change_basis()
    # ghf_gauss.change_basis()

    # for i in range(0, tdho.l, 2):
    #     plt.figure()
    #     plt.title(f"Analytic {i}")
    #     plt.contourf(X, Y, np.abs(gos.spf[i]) ** 2)

    # plt.show()
