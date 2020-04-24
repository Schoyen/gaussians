import numpy as np


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


def overlap(a, l1, A, b, l2, B):
    """Code from Joshua Goings:
    https://joshuagoings.com/2017/04/28/integrals/

    Evaluates overlap integral between two Gaussians
    Returns a float.
    a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
          for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B:    list containing origin of Gaussian 'b'
    """
    S1 = E(l1, l2, 0, A - B, a, b)  # X
    return S1 * np.power(np.pi / (a + b), 0.5)
