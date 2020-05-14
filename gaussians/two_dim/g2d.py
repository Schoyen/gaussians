import numpy as np

from gaussians import G1D


class G2D:
    def __init__(self, alpha: (int, int), a: float, A=[0, 0]):
        assert len(alpha) == len(A) == 2

        self.alpha = alpha
        self.a = a
        self.A = np.asarray(A)

        self.G_x = G1D(self.alpha[0], a, self.A[0], symbol="x")
        self.G_y = G1D(self.alpha[1], a, self.A[1], symbol="y")

        self.norm = self.G_x.norm * self.G_y.norm

    def __call__(
        self, x: np.ndarray, y: np.ndarray, with_norm=False
    ) -> np.ndarray:

        return self.G_x(x, with_norm=with_norm) * self.G_y(
            y, with_norm=with_norm
        )

    def __str__(self):
        return str(self.G_x) + " * " + str(self.G_y)
