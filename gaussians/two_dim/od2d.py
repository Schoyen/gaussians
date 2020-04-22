import numpy as np

from gaussians import OD1D


class OD2D:
    def __init__(self, G_a, G_b):
        self.G_a = G_a
        self.G_b = G_b

        self.O_x = OD1D(self.G_a.G_x, self.G_b.G_x)
        self.O_y = OD1D(self.G_a.G_y, self.G_b.G_y)

    def __call__(self, x, y):
        return self.O_x(x) * self.O_y(x)

    def E(self, t, u):
        return self.O_x.E(t) * self.O_y.E(u)
