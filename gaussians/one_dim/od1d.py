import numpy as np


class OD1D:
    def __init__(self, G_i, G_j):
        self.G_i = G_i
        self.G_j = G_j

        self.i = self.G_i.i
        self.j = self.G_j.i

        # Total exponent
        self.p = self.G_i.a + self.G_j.a
        # Relative separation
        self.X_AB = self.G_i.A - self.G_j.A
        # Center of mass
        self.P = (self.G_i.a * self.G_i.A + self.G_j.a * self.G_j.A) / self.p
        self.X_PA = self.P - self.G_i.A
        self.X_PB = self.P - self.G_j.A
        # Reduced exponent
        self.mu = self.G_i.a * self.G_j.a / self.p

        self.K_AB = np.exp(-self.mu * self.X_AB ** 2)

        self.coefficients = {(0, 0, 0): self.K_AB}

        self.norm = self.G_i.norm * self.G_j.norm

    def __call__(self, x, with_norm=False):
        return self.G_i(x, with_norm=with_norm) * self.G_j(
            x, with_norm=with_norm
        )

    def E(self, t):
        r"""
        >>> from gaussians import G1D
        >>> g_0 = G1D(0, 1.0, -0.5)
        >>> g_1 = G1D(1, 0.7, 0.5)
        >>> g_2 = G1D(2, 1.2, 0.3)
        >>> od_01 = OD1D(g_0, g_1)
        >>> od_02 = OD1D(g_0, g_2)
        >>> od_21 = OD1D(g_2, g_1)
        >>> od_01.E(0) # doctest: +ELLIPSIS
        -0.38969419...
        >>> od_01.E(1) # doctest: +ELLIPSIS
        0.19484709...
        >>> od_01.E(2)
        0
        >>> od_02.E(0) # doctest: +ELLIPSIS
        0.25356869...
        >>> od_02.E(1) # doctest: +ELLIPSIS
        -0.11658330...
        >>> od_21.E(0) # doctest: +ELLIPSIS
        0.00476926...
        >>> od_21.E(2) # doctest: +ELLIPSIS
        0.00143238...
        """
        return self._E(self.i, self.j, t)

    def _E(self, i, j, t):
        r"""This function is not symmetric with respect to interchange
        of ``i`` and ``j``! That is, :math:`E^{ij}_{t} \neq E^{ji}_{t}` in
        general.

        Verify this...
        """
        if (i, j, t) in self.coefficients:
            return self.coefficients[i, j, t]

        if t < 0 or t > (i + j) or i < 0 or j < 0:
            return 0

        if i == 0:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self._E(i, j - 1, t - 1)
                + self.X_PB * self._E(i, j - 1, t)
                + (t + 1) * self._E(i, j - 1, t + 1)
            )
        else:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self._E(i - 1, j, t - 1)
                + self.X_PA * self._E(i - 1, j, t)
                + (t + 1) * self._E(i - 1, j, t + 1)
            )

        return self.coefficients[i, j, t]
