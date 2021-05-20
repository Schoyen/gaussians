import numpy as np
from scipy.special import factorial2


class G1D:
    r"""
    Parameters
    ----------
    i : int
        Angular moment, i.e., the quantum number of the cartesian monomial.
    a : float
        Coefficient in exponent.
    A : float
        Center of Gaussian.
    symbol: str
        Symbol to use when printing the class. Default is ``x``.
    """

    def __init__(self, i: int, a: float, A=0, symbol="x"):
        assert i >= 0

        self.i = i
        self.a = a
        self.A = A
        self.symbol = symbol

        self.norm = 1 / self.compute_norm()

    def compute_norm(self) -> float:
        r"""
        >>> G_0 = G1D(0, 1, 0.0)
        >>> G_0.compute_norm() # doctest: +ELLIPSIS
        1.11951...

        Returns
        -------
        float
            The norm of the one-dimensional Cartesian Gaussian.
        """

        return np.sqrt(
            factorial2(2 * self.i - 1, exact=True)
            / (4 * self.a) ** self.i
            * np.sqrt(np.pi / (2 * self.a))
        )

    def __call__(self, x: np.ndarray, with_norm=False) -> np.ndarray:
        r"""

        Parameters
        ----------
        x : np.ndarray
            Grid points to evaluate the function at.

        Returns
        -------
        np.ndarray
            Value of the primitive Gaussian at the specified points.

        >>> import numpy as np
        >>> x = np.linspace(-2, 2, 101)
        >>> a = 1
        >>> A = -0.5
        >>> G_0 = G1D(0, a, A)
        >>> G_1 = G1D(1, a, A)
        >>> np.allclose(G_0(x) * (x - A), G_1(x))
        True
        >>> G_2 = G1D(2, a, A)
        >>> G_3 = G1D(3, a, A)
        >>> np.allclose(G_2(x) * (x - A), G_3(x))
        True
        """
        x_A = x - self.A
        norm = self.norm if with_norm else 1

        return norm * x_A ** self.i * np.exp(-self.a * x_A ** 2)

    def __str__(self):
        return (
            f"({self.symbol} - {self.A}) ** {self.i} * exp(-{self.a} * "
            + f"({self.symbol} - {self.A}) ** 2)"
        )

    def get_params(self):
        return (self.i, self.a, self.A, self.symbol)

    def decrement_i(self):
        return G1D(self.i - 1, self.a, self.A, self.symbol)

    def increment_i(self):
        return G1D(self.i + 1, self.a, self.A, self.symbol)
