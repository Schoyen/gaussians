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
    construct_coulomb_interaction_matrix_elements,
    construct_kinetic_matrix_elements,
    construct_overlap_matrix_elements,
    construct_multipole_moment_matrix_elements,
)
from gaussians.two_dim.coulomb_elements import I_twiddle

import gaussians.two_dim_lib as tdl

import helpers.coulomb_elements


def test_rec_int():
    t = 0
    u = 0

    p = 1
    sigma = np.random.random(2)

    assert (
        abs(
            abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
            - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
        )
        < 1e-12
    )

    t = 0
    u = 1

    p = 1
    sigma = np.random.random(2)

    assert (
        abs(
            abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
            - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
        )
        < 1e-12
    )

    t = 1
    u = 0

    p = 1
    sigma = np.random.random(2)

    assert (
        abs(
            abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
            - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
        )
        < 1e-12
    )

    t = 1
    u = 1

    p = 1
    sigma = np.random.random(2)

    assert (
        abs(
            abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
            - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
        )
        < 1e-12
    )

    t = 1
    u = 2

    p = 1
    sigma = np.random.random(2)

    assert (
        abs(
            abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
            - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
        )
        < 1e-12
    )

    t = 2
    u = 1

    p = 1
    sigma = np.random.random(2)

    assert (
        abs(
            abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
            - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
        )
        < 1e-12
    )

    for i in range(10):
        t = np.random.randint(5)
        u = np.random.randint(5)

        p = np.random.random() + 0.1
        sigma = np.random.random(2)

        print(i)
        assert (
            abs(
                abs(helpers.coulomb_elements.I_tilde(t, u, p, sigma))
                - abs(I_twiddle(t, u, 1 / (4 * p), sigma))
            )
            < 1e-12
        )


def test_construction():
    gaussians = [
        G2D((0, 0), 1, (0, 0.5)),
        G2D((1, 0), 1, (-0.3, -0.5)),
        G2D((0, 1), 1),
        G2D((1, 1), 1, (0.4, -0.2)),
    ]

    u = construct_coulomb_interaction_matrix_elements(gaussians)
    u_2 = (
        helpers.coulomb_elements.construct_coulomb_interaction_matrix_elements(
            gaussians
        )
    )
    u_r = tdl.construct_coulomb_interaction_operator_matrix_elements(
        [g.get_params() for g in gaussians]
    )

    np.testing.assert_allclose(u, u_2)
    np.testing.assert_allclose(u, u_r)
    np.testing.assert_allclose(u, u.transpose(2, 3, 0, 1))
    np.testing.assert_allclose(u, u.transpose(2, 1, 0, 3))
    np.testing.assert_allclose(u, u.transpose(0, 3, 2, 1))


def test_two_dim_ho():
    omega = 1
    l = 6
    n = 2

    shell = l // 2 + l % 2

    gaussians = []
    for i in range(shell):
        for j in range(i, shell - i):
            gaussians.append(G2D((i, j), omega / 2))

            if i != j:
                gaussians.append(G2D((j, i), omega / 2))

    for gauss in gaussians:
        print(gauss)

    tdho = GeneralOrbitalSystem(
        n, TwoDimensionalHarmonicOscillator(l, 5, 201, omega=omega)
    )

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
    u = construct_coulomb_interaction_matrix_elements(gaussians)
    u_2 = (
        helpers.coulomb_elements.construct_coulomb_interaction_matrix_elements(
            gaussians
        )
    )
    np.testing.assert_allclose(u, u_2)

    gos = BasisSet(len(gaussians), dim=2)
    gos.h = h
    gos.s = s
    gos.u = u

    spf = np.asarray([gauss(X, Y, with_norm=True) for gauss in gaussians])
    gos.spf = spf

    gos = GeneralOrbitalSystem(n, gos)

    # eps, C = scipy.linalg.eigh(gos.h, gos.s)
    # print(eps)
    # print(np.diag(tdho.h))
    # print(C)
    # import matplotlib.pyplot as plt
    # plt.imshow(C)

    # gos.change_basis(C)

    # plt.show()
    # wat
    #

    ghf_ho = GHF(tdho, verbose=True)
    ghf_ho.compute_ground_state(tol=1e-8)
    # ghf_ho = HartreeFock(tdho, verbose=True)
    # ghf_ho.scf()

    ghf_gauss = GHF(gos, verbose=True)
    ghf_gauss.compute_ground_state(tol=1e-8)
    # ghf_gauss = HartreeFock(gos, verbose=True)
    # ghf_gauss.scf()

    print(ghf_ho.epsilon)
    print(ghf_gauss.epsilon)
    assert abs(ghf_ho.compute_energy() - ghf_gauss.compute_energy()) < 1e-10
    # assert abs(ghf_ho.e_hf - ghf_gauss.e_hf) < 1e-12

    # import matplotlib.pyplot as plt

    # u_tdho = tdho.u.reshape(tdho.l ** 2, tdho.l ** 2)
    # u_gauss = gos.u.reshape(gos.l ** 2, gos.l ** 2)

    # plt.figure()
    # plt.imshow(u_tdho)
    # plt.title("TDHO")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(u_gauss)
    # plt.title("Gauss")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(u_tdho - u_gauss)
    # plt.title("Diff")
    # plt.colorbar()

    # gos_2 = gos.copy_system()

    # eps, C = scipy.linalg.eigh(gos_2.h, gos_2.s)
    # gos_2.change_basis(C)

    # u_gauss = gos_2.u.reshape(gos_2.l ** 2, gos_2.l ** 2)

    # plt.figure()
    # plt.imshow(u_gauss)
    # plt.title("Gauss (HO)")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(u_tdho - u_gauss)
    # plt.title("Diff (HO)")
    # plt.colorbar()

    # np.testing.assert_allclose(gos_2.h, tdho.h, atol=1e-14)
    # np.testing.assert_allclose(gos_2.s, tdho.s, atol=1e-14)

    # tdho.change_basis(ghf_ho.C)
    # gos.change_basis(ghf_gauss.C)

    # u_tdho = tdho.u.reshape(tdho.l ** 2, tdho.l ** 2)
    # u_gauss = gos.u.reshape(gos.l ** 2, gos.l ** 2)

    # plt.figure()
    # plt.imshow(u_tdho)
    # plt.title("TDHO (HF)")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(u_gauss)
    # plt.title("Gauss (HF)")
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(u_tdho - u_gauss)
    # plt.title("Diff (HF)")
    # plt.colorbar()

    # np.testing.assert_allclose(ghf_ho.epsilon, ghf_gauss.epsilon)

    # for i in range(0, tdho.l, 2):
    #     plt.figure()
    #     plt.title(f"Grid: {i}")
    #     plt.contourf(X, Y, np.abs(tdho.spf[i]) ** 2)

    # for i in range(0, tdho.l, 2):
    #     plt.figure()
    #     plt.title(f"Analytic {i}")
    #     plt.contourf(X, Y, np.abs(gos.spf[i]) ** 2)

    # for i in range(0, tdho.l, 2):
    #     plt.figure()
    #     plt.title(f"Grid: {i}")
    #     plt.contourf(X, Y, np.abs(tdho.spf[i]) ** 2)

    # for i in range(0, tdho.l, 2):
    #     plt.figure()
    #     plt.title(f"Analytic {i}")
    #     plt.contourf(X, Y, np.abs(gos.spf[i]) ** 2)

    # rho_qp = np.zeros((tdho.l, tdho.l))
    # rho_qp[0, 0] = 1
    # rho_qp[1, 1] = 1

    # C = np.eye(tdho.l)

    # rho_ho = tdho.compute_particle_density(rho_qp, C)
    # rho_gauss = gos.compute_particle_density(rho_qp, C)
    # # rho_ho = ghf_ho.compute_particle_density()
    # # rho_gauss = ghf_gauss.compute_particle_density()

    # plt.figure()
    # plt.title("HO particle density")
    # plt.contourf(X, Y, np.abs(rho_ho) ** 2)

    # plt.figure()
    # plt.title("Gauss particle density")
    # plt.contourf(X, Y, np.abs(rho_ho) ** 2)

    # plt.show()
