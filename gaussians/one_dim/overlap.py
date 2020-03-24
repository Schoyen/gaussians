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
        else:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self.E(i - 1, j, t - 1)
                - self.mu * self.X_AB / self.G_i.a * self.E(i - 1, j, t)
                + (t + 1) * self.E(i - 1, j, t + 1)
            )

        return self.coefficients[i, j, t]


def overlap(G_i, G_j):
    r"""Function computing the overlap between two one-dimensional primitive
    GTO's. This is given by

    .. math:: s^{i}_{j} = \langle G_i(a, A) \rvert G_j(b, B) \rangle
        = E^{ij}_{0} \sqrt{\frac{\pi}{p}},

    where :math:`p = a + b` as the total exponent of the Gaussians, :math:`A`
    and :math:`B` are the centers of the GTO's, and :math:`E^{ij}_{0}` are the
    Hermite expansion coefficients.

    >>> import numpy as np
    >>> from gaussians.one_dim.g1d import G1D
    >>> G_0 = G1D(0, 1, 0)
    >>> overlap(G_0, G_0) # doctest.ELLIPSIS
    0.99999999...
    >>> G_1 = G1D(1, 1, 0)
    >>> overlap(G_1, G_1) # doctest.ELLIPSIS
    0.99999999...
    """
    omega_ij = OverlapDist(G_i, G_j)

    return (
        G_i.norm
        * G_j.norm
        * omega_ij.E(G_i.i, G_j.i, 0)
        * np.sqrt(np.pi / omega_ij.p)
    )


def construct_overlap_matrix(gaussians):
    r"""

    >>> from gaussians.one_dim.g1d import G1D
    >>> s = construct_overlap_matrix([G1D(0, 2, -4), G1D(0, 2, 4)])
    >>> s.shape
    (2, 2)
    >>> abs(s[0, 0] - s[1, 1]) < 1e-12
    True
    >>> abs(s[0, 1] - s[1, 0]) < 1e-12
    True
    >>> abs(s[0, 1]) < 1e-12
    True
    """

    l = len(gaussians)
    s = np.zeros((l, l))

    for i, G_i in enumerate(gaussians):
        for j, G_j in enumerate(gaussians):
            s[i, j] = overlap(G_i, G_j)

    return s
