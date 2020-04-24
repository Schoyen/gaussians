import numpy as np

from gaussians import G1D, OD1D


def E(i, j, t, Qx, a, b):
    """Code from Joshua Goings:
    https://joshuagoings.com/2017/04/28/integrals/

    Recursive definition of Hermite Gaussian coefficients.
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    i,j: orbital angular momentum number on Gaussian 'a' and 'b'
    t: number nodes in Hermite (depends on type of integral,
       e.g. always zero for overlap integrals)
    Qx: distance between origins of Gaussian 'a' and 'b'
    """
    p = a + b
    q = a * b / p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q * Qx * Qx)  # K_AB
    elif j == 0:
        # decrement index i
        return (
            (1 / (2 * p)) * E(i - 1, j, t - 1, Qx, a, b)
            - (q * Qx / a) * E(i - 1, j, t, Qx, a, b)
            + (t + 1) * E(i - 1, j, t + 1, Qx, a, b)
        )
    else:
        # decrement index j
        return (
            (1 / (2 * p)) * E(i, j - 1, t - 1, Qx, a, b)
            + (q * Qx / b) * E(i, j - 1, t, Qx, a, b)
            + (t + 1) * E(i, j - 1, t + 1, Qx, a, b)
        )


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
                print(t)
                np.testing.assert_allclose(
                    E(O_ij.i, O_ij.j, t, O_ij.X_AB, O_ij.G_i.a, O_ij.G_j.a),
                    O_ij.E(t),
                )
                np.testing.assert_allclose(
                    E(O_ji.i, O_ji.j, t, O_ji.X_AB, O_ji.G_i.a, O_ji.G_j.a),
                    O_ji.E(t),
                )
