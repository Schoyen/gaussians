import numpy as np


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
    """

    def __init__(self, i, a, A=0):
        self.i = i
        self.a = a
        self.A = A

    def __call__(self, x):
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
        >>> G_2 = G1D(2, a, A)
        >>> G_3 = G1D(3, a, A)
        >>> np.allclose(G_2(x) * (x - A), G_3(x))
        True
        """
        x_A = x - self.A

        return x_A ** self.i * np.exp(-self.a * x_A ** 2)
