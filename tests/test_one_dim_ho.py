import pytest

import numpy as np
import scipy.linalg

from gaussians import G1D
from gaussians.one_dim import (
    construct_overlap_matrix_elements,
    construct_kinetic_matrix_elements,
    construct_differential_matrix_elements,
    construct_multipole_moment_matrix_elements,
    construct_shielded_coulomb_interaction_matrix_elements,
)

import gaussians.one_dim_lib as odl

from quantum_systems import ODQD, BasisSet


@pytest.fixture
def odho_hamiltonian():
    # Harmonic oscillator functions are Gaussians located at the center of the
    # well.
    l = 30
    omega = 1

    gaussians = []

    for i in range(l):
        gaussians.append(G1D(i, omega / 2, 0))

    t = -0.5 * construct_differential_matrix_elements(2, gaussians)
    v = (
        0.5
        * omega ** 2
        * construct_multipole_moment_matrix_elements(2, 0, gaussians)
    )
    s = construct_multipole_moment_matrix_elements(0, 0, gaussians)

    return t + v, s, omega


def test_eigenenergies(odho_hamiltonian):
    h, s, omega = odho_hamiltonian

    energies, states = scipy.linalg.eigh(h, s)
    l = len(energies)

    eps = lambda n: omega * (n + 0.5)

    # We only test the first 10 eigenenergies. Higher order will be worse.
    for i in range(10):
        np.testing.assert_allclose(energies[i], eps(i), atol=1e-12)


def test_rust_one_dim_lib():
    import time

    omega = 0.5
    l = 20

    gaussians = [G1D(i, omega / 2, 0) for i in range(l)]

    t_0 = time.time()
    t = construct_kinetic_matrix_elements(gaussians)
    v = (
        0.5
        * omega ** 2
        * construct_multipole_moment_matrix_elements(2, 0, gaussians)
    )
    s = construct_overlap_matrix_elements(gaussians)
    t_1 = time.time()
    print(f"Time Python: {t_1 - t_0} sec")
    h = t + v

    g_params = [g.get_params() for g in gaussians]

    t_0 = time.time()
    t_r = odl.construct_kinetic_operator_matrix_elements(g_params)
    v_r = (
        0.5
        * omega ** 2
        * odl.construct_multipole_moment_matrix_elements(2, 0, g_params)
    )
    s_r = odl.construct_overlap_matrix_elements(g_params)
    t_1 = time.time()
    print(f"Time Rust: {t_1 - t_0} sec")
    h_r = t_r + v_r

    np.testing.assert_allclose(t, t_r)
    np.testing.assert_allclose(v, v_r)
    np.testing.assert_allclose(s, s_r)
    np.testing.assert_allclose(h, h_r)


def test_odho_qs_comparison():
    l = 6
    omega = 0.5
    grid_length = 6
    num_grid_points = 1001

    a = 0.25
    alpha = 1.0

    odho = ODQD(
        l,
        grid_length,
        num_grid_points,
        a=a,
        alpha=alpha,
        potential=ODQD.HOPotential(omega),
    )

    grid = odho.grid

    gaussians = [G1D(i, omega / 2, 0) for i in range(l)]

    t = -0.5 * construct_differential_matrix_elements(2, gaussians)
    v = (
        0.5
        * omega ** 2
        * construct_multipole_moment_matrix_elements(2, 0, gaussians)
    )
    s = construct_overlap_matrix_elements(gaussians)
    spf = np.asarray([g(grid, with_norm=True) for g in gaussians])
    spf[:, 0] = 0
    spf[:, -1] = 0
    u = construct_shielded_coulomb_interaction_matrix_elements(
        spf, grid, alpha, a
    )

    odg = BasisSet(l, dim=1)
    odg.h = t + v
    odg.s = s
    odg.spf = spf
    odg.u = u

    eps, C = scipy.linalg.eigh(odg.h, odg.s)

    odg.change_basis(C)

    import matplotlib.pyplot as plt

    for i in range(l):
        plt.plot(grid, odho.spf[i].real, label=rf"odho {i}")
        plt.plot(grid, odg.spf[i].real, "--", label=rf"odg {i}")

    plt.grid()
    plt.legend()
    plt.show()

    np.testing.assert_allclose(odho.h, odg.h, atol=1e-2)
    # np.testing.assert_allclose(odho.u, odg.u, atol=1e-2)


if __name__ == "__main__":
    test_odho_qs_comparison()
