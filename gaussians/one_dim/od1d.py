import numpy as np


class OD1D:
    def __init__(self, G_i, G_j):
        self.G_i = G_i
        self.G_j = G_j

        self.i = self.G_i.i
        self.j = self.G_i.i

        # Total exponent
        self.p = self.G_i.a + self.G_j.a
        # Relative separation
        self.X_AB = self.G_i.A - self.G_j.A
        # Center of mass
        self.P = (self.G_i.a * self.G_i.A + self.G_j.a * self.G_j.A) / self.p
        # Reduced exponent
        self.mu = self.G_i.a * self.G_j.a / (self.G_i.a + self.G_j.a)

        self.K_AB = np.exp(-self.mu * self.X_AB ** 2)

        self.coefficients = {(0, 0, 0): self.K_AB}

        self.norm = self.G_i.norm * self.G_j.norm

    def __call__(self, x, with_norm=False):
        return self.G_i(x, with_norm=with_norm) * self.G_j(
            x, with_norm=with_norm
        )

    def E(self, t):
        return self._E(self.i, self.j, t)

    def _E(self, i, j, t):
        if (i, j, t) in self.coefficients:
            return self.coefficients[i, j, t]

        if t < 0 or t > i + j:
            return 0

        if i == 0:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self._E(i, j - 1, t - 1)
                + self.mu * self.X_AB / self.G_j.a * self._E(i, j - 1, t)
                + (t + 1) * self._E(i, j - 1, t + 1)
            )
        else:
            self.coefficients[i, j, t] = (
                1 / (2 * self.p) * self._E(i - 1, j, t - 1)
                - self.mu * self.X_AB / self.G_i.a * self._E(i - 1, j, t)
                + (t + 1) * self._E(i - 1, j, t + 1)
            )

        return self.coefficients[i, j, t]
