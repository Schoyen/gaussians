import numpy as np

from gaussians import G1D, OD1D
from gaussians.one_dim.multipole_moment import (
    construct_multipole_moment_matrix_elements,
)

from helpers.utils import overlap as j_overlap, E


def overlap(G_i, G_j):
    O_ij = OD1D(G_i, G_j)

    return O_ij.E(0) * np.sqrt(np.pi / O_ij.p)


def construct_overlap_matrix(gaussians):
    l = len(gaussians)
    s = np.zeros((l, l))

    for i in range(l):
        G_i = gaussians[i]

        for j in range(l):
            G_j = gaussians[j]

            s[i, j] = G_i.norm * G_j.norm * overlap(G_i, G_j)

    return s


def dipole_moment(C, G_i, G_j):
    O_ij = OD1D(G_i, G_j)

    X_PC = O_ij.P - C
    p = O_ij.p

    return (O_ij.E(1) + X_PC * O_ij.E(0)) * np.sqrt(np.pi / p)


def construct_dipole_moment_matrix(gaussians):
    l = len(gaussians)
    d = np.zeros((l, l))

    for i in range(l):
        G_i = gaussians[i]

        for j in range(l):
            G_j = gaussians[j]

            d[i, j] = G_i.norm * G_j.norm * dipole_moment(1, G_i, G_j)

    return d


def test_overlap():
    gaussians = [G1D(0, 1, 0.5), G1D(0, 0.5, 0), G1D(1, 1, 0), G1D(2, 1, 0)]

    s = construct_multipole_moment_matrix_elements(0, 0, gaussians)
    s_2 = construct_overlap_matrix(gaussians)

    x = np.linspace(-10, 10, 1001)

    s_3 = np.zeros_like(s_2)
    s_4 = np.zeros_like(s_2)

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            s_3[i, j] = G_i.norm * G_j.norm * np.trapz(G_i(x) * G_j(x), x=x)
            s_4[i, j] = (
                G_i.norm
                * G_j.norm
                * j_overlap(G_i.a, G_i.i, G_i.A, G_j.a, G_j.i, G_j.A)
            )

    np.testing.assert_allclose(s, s_2)
    np.testing.assert_allclose(s_3, s_4, atol=1e-12)
    np.testing.assert_allclose(s, s_4)
    np.testing.assert_allclose(s, s_3, atol=1e-12)
    np.testing.assert_allclose(s_2, s_2.T)
    np.testing.assert_allclose(s, s.T)


def test_dipole_moment():
    gaussians = [G1D(0, 2, -4), G1D(0, 2, 4), G1D(1, 1, 0), G1D(2, 1, 0)]

    d = construct_multipole_moment_matrix_elements(1, 1, gaussians)
    d_2 = construct_dipole_moment_matrix(gaussians)

    np.testing.assert_allclose(d, d_2)
    np.testing.assert_allclose(d, d.T)
