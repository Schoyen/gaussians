import numpy as np
import scipy.sparse


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


def two_dim_grid_solver(potential):
    N = 50
    Lx = 5
    x = np.linspace(-Lx, Lx, N + 2)
    y = np.linspace(-Lx, Lx, N + 2)
    V = np.zeros((N, N))
    delta_x = x[1] - x[0]
    w = 1
    R = 1.5

    X, Y = np.meshgrid(x[1 : N + 1], y[1 : N + 1])
    V = potential(X, Y)
    V = V.T

    n = N**2
    a = 0.5

    h_diag = a * 4 * np.ones(n) / (delta_x**2) + V.flatten("F")
    h_off = -a * np.ones(n - 1) / (delta_x**2)
    h_off_off = -a * np.ones(n - N) / (delta_x**2)

    k = 1
    for i in range(1, n - 1):
        if i % N == 0:
            h_off[i - 1] = 0

    h = scipy.sparse.diags(
        [h_diag, h_off, h_off, h_off_off, h_off_off], offsets=[0, -1, 1, -N, N]
    )

    h = h.todense()
    epsilon, phi = np.linalg.eigh(h)

    return epsilon, phi, X, Y
