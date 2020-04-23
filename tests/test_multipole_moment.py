import numpy as np

from gaussians import G1D, OD1D
from gaussians.one_dim.multipole_moment import construct_multipole_moment_matrix


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


def test_overlap_dist():
    gaussians = [G1D(0, 2, -4), G1D(0, 2, 4), G1D(1, 1, 0), G1D(2, 1, 0)]

    for G_i in gaussians:
        for G_j in gaussians:
            O_ij = OD1D(G_i, G_j)
            O_ji = OD1D(G_j, G_i)

            for t in range(5):
                print(t)
                np.testing.assert_allclose(O_ij.E(t), O_ji.E(t))


def test_overlap():
    gaussians = [G1D(0, 2, -4), G1D(0, 2, 4), G1D(1, 1, 0), G1D(2, 1, 0)]

    s = construct_multipole_moment_matrix(0, 0, gaussians)
    s_2 = construct_overlap_matrix(gaussians)

    np.testing.assert_allclose(s, s_2)
    np.testing.assert_allclose(s_2, s_2.T)
    np.testing.assert_allclose(s, s.T)


def test_dipole_moment():
    gaussians = [G1D(0, 2, -4), G1D(0, 2, 4), G1D(1, 1, 0), G1D(2, 1, 0)]

    d = construct_multipole_moment_matrix(1, 1, gaussians)
    d_2 = construct_dipole_moment_matrix(gaussians)

    np.testing.assert_allclose(d, d_2)
