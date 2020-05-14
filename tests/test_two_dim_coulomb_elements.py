import numpy as np
import scipy.linalg

from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    BasisSet,
    GeneralOrbitalSystem,
)

from hartree_fock import GHF

# from tdhf import HartreeFock

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
    l = 6
    n = 2

    shell = l // 2

    gaussians = []
    for i in range(shell):
        for j in range(i, shell - i):
            gaussians.append(G2D((i, j), omega / 2))

            if i != j:
                gaussians.append(G2D((j, i), omega / 2))

    tdho = GeneralOrbitalSystem(
        n, TwoDimensionalHarmonicOscillator(6, 5, 201, omega=omega)
    )

    ghf_ho = GHF(tdho, verbose=True)
    ghf_ho.compute_ground_state()
    # ghf_ho = HartreeFock(tdho, verbose=True)
    # ghf_ho.scf()

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

    gos = GeneralOrbitalSystem(n, gos)

    ghf_gauss = GHF(gos, verbose=True)
    ghf_gauss.compute_ground_state()
    # ghf_gauss = HartreeFock(gos, verbose=True)
    # ghf_gauss.scf()

    print(ghf_ho.epsilon)
    print(ghf_gauss.epsilon)
    assert abs(ghf_ho.compute_energy() - ghf_gauss.compute_energy()) < 1e-12
    # assert abs(ghf_ho.e_hf - ghf_gauss.e_hf) < 1e-12

    import matplotlib.pyplot as plt

    for i in range(0, tdho.l, 2):
        plt.figure()
        plt.title(f"Grid: {i}")
        plt.contourf(X, Y, np.abs(tdho.spf[i]) ** 2)

    for i in range(0, tdho.l, 2):
        plt.figure()
        plt.title(f"Analytic {i}")
        plt.contourf(X, Y, np.abs(gos.spf[i]) ** 2)

    plt.show()

    wat

    # ghf_ho.change_basis()
    # ghf_gauss.change_basis()
    tdho.change_basis(ghf_ho.C)
    gos.change_basis(ghf_gauss.C)

    for i in range(0, tdho.l, 2):
        plt.figure()
        plt.title(f"Grid: {i}")
        plt.contourf(X, Y, np.abs(tdho.spf[i]) ** 2)

    for i in range(0, tdho.l, 2):
        plt.figure()
        plt.title(f"Analytic {i}")
        plt.contourf(X, Y, np.abs(gos.spf[i]) ** 2)

    rho_qp = np.zeros((thdo.l, tdho.l))
    rho_qp[0, 0] = 1
    rho_qp[1, 1] = 1

    C = np.eye(tdho.l)

    rho_ho = tdho.compute_particle_density(rho_qp, C)
    rho_gauss = gos.compute_particle_density(rho_qp, C)
    # rho_ho = ghf_ho.compute_particle_density()
    # rho_gauss = ghf_gauss.compute_particle_density()

    plt.figure()
    plt.title("HO particle density")
    plt.contourf(X, Y, np.abs(rho_ho) ** 2)

    plt.figure()
    plt.title("Gauss particle density")
    plt.contourf(X, Y, np.abs(rho_ho) ** 2)

    plt.show()
