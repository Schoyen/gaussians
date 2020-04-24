import numpy as np

from gaussians import G1D, OD1D

from helpers.utils import E


def test_hermite_expansion_symmetry():
    np.random.seed(2020)
    angular_mom = [0, 1, 2, 3, 4]

    gaussians = list()

    for l in angular_mom:
        g1d = G1D(
            l,
            np.random.uniform(low=0.1, high=4),
            np.random.uniform(low=-2, high=2),
        )
        gaussians.append(g1d)

    for G_i in gaussians:
        for G_j in gaussians:
            O_ij = OD1D(G_i, G_j)
            O_ji = OD1D(G_j, G_i)

            np.testing.assert_allclose(O_ij.p, O_ji.p)
            np.testing.assert_allclose(O_ij.X_AB, -O_ji.X_AB)
            np.testing.assert_allclose(O_ij.mu, O_ji.mu)
            np.testing.assert_allclose(O_ij.P, O_ji.P)
            np.testing.assert_allclose(O_ij.norm, O_ji.norm)
            np.testing.assert_allclose(O_ij.norm, G_i.norm * G_j.norm)
            np.testing.assert_allclose(O_ij.K_AB, O_ji.K_AB)

            for t in range(max(angular_mom)):
                np.testing.assert_allclose(
                    E(O_ij.i, O_ij.j, t, O_ij.X_AB, O_ij.G_i.a, O_ij.G_j.a),
                    O_ij.E(t),
                )
                np.testing.assert_allclose(
                    E(O_ji.i, O_ji.j, t, O_ji.X_AB, O_ji.G_i.a, O_ji.G_j.a),
                    O_ji.E(t),
                )
