import pytest

import numpy as np
import scipy.linalg

from gaussians import G1D
from gaussians.one_dim import (
    construct_differential_matrix_elements,
    construct_multipole_moment_matrix_elements,
)


@pytest.fixture
def odho_hamiltonian():
    # Harmonic oscillator functions are Gaussians located at the center of the
    # well.
    l = 30
    omega = 1

    gaussians = []

    for i in range(l):
        gaussians.append(G1D(i, omega / np.sqrt(2), 0))

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
        np.testing.assert_allclose(energies[i], eps(i), atol=1e-7)
