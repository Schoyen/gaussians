import math

import numpy as np
import scipy

from quantum_systems import (
    TwoDimensionalHarmonicOscillator,
    BasisSet,
    GeneralOrbitalSystem,
)


from configuration_interaction import CISD


from gaussians import G2D
from gaussians.two_dim import (
    construct_coulomb_interaction_matrix_elements,
    construct_kinetic_matrix_elements,
    construct_overlap_matrix_elements,
    construct_multipole_moment_matrix_elements,
)

import gaussians.two_dim_lib as tdl


def one_dim_ho_spf(n, x, omega):
    return (
        1
        / np.sqrt(2 ** n * math.factorial(n))
        * (omega / np.pi) ** (1 / 4)
        * np.exp(-omega / 2 * x ** 2)
        * scipy.special.eval_hermite(n, x)
    )


def test_off_centered_ho():
    omega = 1
    n = 2
    l = 6

    center = (1.0, -0.5)

    gaussians = [
        G2D((0, 0), omega / 2, center),
        G2D((0, 1), omega / 2, center),
        G2D((1, 0), omega / 2, center),
        G2D((2, 0), omega / 2, center),
        G2D((1, 1), omega / 2, center),
        G2D((0, 2), omega / 2, center),
    ]
    g_params = [g.get_params() for g in gaussians]

    tdho = TwoDimensionalHarmonicOscillator(l, 6, 401, omega=omega)

    X = tdho.R * np.cos(tdho.T)
    Y = tdho.R * np.sin(tdho.T)

    t = construct_kinetic_matrix_elements(gaussians)
    v = (
        0.5
        * omega ** 2
        * (
            construct_multipole_moment_matrix_elements(
                [2, 0], center, gaussians
            )
            + construct_multipole_moment_matrix_elements(
                [0, 2], center, gaussians
            )
        )
    )

    h = t + v
    s = construct_overlap_matrix_elements(gaussians)
    u = construct_coulomb_interaction_matrix_elements(gaussians)

    t_r = tdl.construct_kinetic_operator_matrix_elements(g_params)
    v_r = (
        0.5
        * omega ** 2
        * (
            tdl.construct_multipole_moment_matrix_elements(
                (2, 0), center, g_params
            )
            + tdl.construct_multipole_moment_matrix_elements(
                (0, 2), center, g_params
            )
        )
    )
    h_r = t_r + v_r
    s_r = tdl.construct_overlap_matrix_elements(g_params)
    u_r = tdl.construct_coulomb_interaction_operator_matrix_elements(g_params)

    np.testing.assert_allclose(t, t_r, atol=1e-14)
    np.testing.assert_allclose(v, v_r)
    np.testing.assert_allclose(h, h_r)
    np.testing.assert_allclose(s, s_r)
    np.testing.assert_allclose(u, u_r)

    bs = BasisSet(len(gaussians), dim=2)
    bs.h = h_r
    bs.s = s_r
    bs.u = u_r

    bs.spf = np.asarray([gauss(X, Y, with_norm=True) for gauss in gaussians])

    eps, C = scipy.linalg.eigh(bs.h, bs.s)

    bs.change_basis(C=C)

    np.testing.assert_allclose(bs.h, tdho.h, atol=1e-12)
    np.testing.assert_allclose(bs.s, np.eye(bs.l), atol=1e-12)

    tdho_gos = GeneralOrbitalSystem(n, tdho)
    gauss_gos = GeneralOrbitalSystem(n, bs)

    tdho_ci = CISD(tdho_gos, verbose=True).compute_ground_state()
    gauss_ci = CISD(gauss_gos, verbose=True).compute_ground_state()

    np.testing.assert_allclose(
        tdho_ci.compute_energy(), gauss_ci.compute_energy()
    )

    np.testing.assert_allclose(tdho_ci.energies, gauss_ci.energies)


def test_two_dim_ho():
    omega = 1
    n = 2
    l = 6

    gaussians = [
        G2D((0, 0), omega / 2),
        G2D((0, 1), omega / 2),
        G2D((1, 0), omega / 2),
        G2D((2, 0), omega / 2),
        G2D((1, 1), omega / 2),
        G2D((0, 2), omega / 2),
    ]
    g_params = [g.get_params() for g in gaussians]

    tdho = TwoDimensionalHarmonicOscillator(l, 6, 401, omega=omega)

    X = tdho.R * np.cos(tdho.T)
    Y = tdho.R * np.sin(tdho.T)

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

    t_r = tdl.construct_kinetic_operator_matrix_elements(g_params)
    v_r = (
        0.5
        * omega ** 2
        * (
            tdl.construct_multipole_moment_matrix_elements(
                (2, 0), (0, 0), g_params
            )
            + tdl.construct_multipole_moment_matrix_elements(
                (0, 2), (0, 0), g_params
            )
        )
    )
    h_r = t_r + v_r
    s_r = tdl.construct_overlap_matrix_elements(g_params)
    u_r = tdl.construct_coulomb_interaction_operator_matrix_elements(g_params)

    np.testing.assert_allclose(t, t_r, atol=1e-14)
    np.testing.assert_allclose(v, v_r)
    np.testing.assert_allclose(h, h_r)
    np.testing.assert_allclose(s, s_r)
    np.testing.assert_allclose(u, u_r)

    bs = BasisSet(len(gaussians), dim=2)
    bs.h = h
    bs.s = s
    bs.u = u

    bs.spf = np.asarray([gauss(X, Y, with_norm=True) for gauss in gaussians])

    eps, C = scipy.linalg.eigh(bs.h, bs.s)

    bs.change_basis(C=C)

    np.testing.assert_allclose(bs.h, tdho.h, atol=1e-12)
    np.testing.assert_allclose(bs.s, np.eye(bs.l), atol=1e-12)

    # Compare single-particle functions to analytical cartesian 2D harmonic
    # oscillator basis functions.
    psi_00 = one_dim_ho_spf(0, X, omega) * one_dim_ho_spf(0, Y, omega)
    np.testing.assert_allclose(
        np.abs(psi_00) ** 2, np.abs(bs.spf[0]) ** 2, atol=1e-12
    )

    psi_10 = one_dim_ho_spf(1, X, omega) * one_dim_ho_spf(0, Y, omega)
    psi_01 = one_dim_ho_spf(0, X, omega) * one_dim_ho_spf(1, Y, omega)
    np.testing.assert_allclose(
        1 / 2 * (np.abs(psi_10) ** 2 + np.abs(psi_01) ** 2),
        1 / 2 * (np.abs(bs.spf[1]) ** 2 + np.abs(bs.spf[2]) ** 2),
        atol=1e-12,
    )

    psi_20 = one_dim_ho_spf(2, X, omega) * one_dim_ho_spf(0, Y, omega)
    psi_02 = one_dim_ho_spf(0, X, omega) * one_dim_ho_spf(2, Y, omega)
    psi_11 = one_dim_ho_spf(1, X, omega) * one_dim_ho_spf(1, Y, omega)
    np.testing.assert_allclose(
        1
        / 3
        * (np.abs(psi_20) ** 2 + np.abs(psi_02) ** 2 + np.abs(psi_11) ** 2),
        1
        / 3
        * (
            np.abs(bs.spf[3]) ** 2
            + np.abs(bs.spf[4]) ** 2
            + np.abs(bs.spf[5]) ** 2
        ),
        atol=1e-12,
    )

    tdho_gos = GeneralOrbitalSystem(n, tdho)
    gauss_gos = GeneralOrbitalSystem(n, bs)

    tdho_ci = CISD(tdho_gos, verbose=True).compute_ground_state()
    gauss_ci = CISD(gauss_gos, verbose=True).compute_ground_state()

    np.testing.assert_allclose(
        tdho_ci.compute_energy(), gauss_ci.compute_energy()
    )

    np.testing.assert_allclose(tdho_ci.energies, gauss_ci.energies)
