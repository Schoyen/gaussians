import numpy as np

from gaussians import OD1D


class OD2D:
    def __init__(self, G_a, G_b):
        self.G_a = G_a
        self.G_b = G_b

        self.O_x = OD1D(self.G_a.G_x, self.G_b.G_x)
        self.O_y = OD1D(self.G_a.G_y, self.G_b.G_y)

        assert self.O_x.p == self.O_y.p
        self.p = self.O_x.p

        self.P = self.G_a.A - self.G_b.A

        self.x_sum_lim = self.O_x.i + self.O_x.j
        self.y_sum_lim = self.O_y.i + self.O_y.j

    def __call__(self, x, y, with_norm=False):
        return self.O_x(x, with_norm=with_norm) * self.O_y(
            x, with_norm=with_norm
        )

    def E(self, t, u):
        return self.O_x.E(t) * self.O_y.E(u)
