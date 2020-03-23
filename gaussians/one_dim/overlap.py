import numpy as np


class OverlapDist:
    def __init__(self, G_i, G_j):
        self.G_i = G_i
        self.G_j = G_j

        # Total exponent
        self.p = self.G_i.a + self.G_j.a
        # Relative separation
        self.X_AB = self.G_i.A - self.G_j.A
        # Center of mass
        self.P_x = (self.G_i.a * self.G_i.A + self.G_j.a * self.G_j.A) / self.p
        # Reduced exponent
        self.mu = self.G_i.a * self.G_j.a / (self.G_i.a + self.G_j.a)

        self.K_AB = np.exp(-self.mu * self.X_AB ** 2)

        self.coefficients = dict()
        self.coefficients[0, 0, 0] = self.K_AB

    def __call__(self, x):
        return self.G_i(x) * self.G_j(x)

    def E(self, i, j, t):
        if (i, j, t) in self.coefficients:
            return self.coefficients[i, j, t]

        if t < 0 or t > i + j:
            return 0

        if i == 0:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self.E(i, j - 1, t - 1)
                + self.mu * self.X_AB / self.G_j.a * self.E(i, j - 1, t)
                + (t + 1) * self.E(i, j - 1, t + 1)
            )

        elif j == 0:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self.E(i - 1, j, t - 1)
                - self.mu * self.X_AB / self.G_i.a * self.E(i - 1, j, t)
                + (t + 1) * self.E(i - 1, j, t + 1)
            )

        return self.coefficients[i, j, t]


def overlap(G_i, G_j):
    r"""
    >>> from gaussians.one_dim.g1d import G1D
    >>> G_0 = G1D(0, 1, 0)
    >>> overlap(G_0, G_0)
    """
    omega_ij = OverlapDist(G_i, G_j)

    return omega_ij.E(G_i.i, G_j.i, 0) * np.sqrt(np.pi / omega_ij.p)
