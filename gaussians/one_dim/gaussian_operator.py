import numpy as np

from .g1d import G1D
from .od1d import OD1D


def construct_gaussian_operator_matrix_elements(
    gaussians: list,
    c: float,
    center=0.0,
    k=0,
) -> np.ndarray:
    r"""Compute the three-center matrix elements between three Gaussian
    orbitals. One of the orbitals is the Gaussian well. The matrix elements are
    thus on the form

    .. math:: G^k_{ij} = \langle g_i(a, A) |
        \hat{G}_k(x; c, C) | g_j(b, B) \rangle,

    where the Gaussian-well operator is given by

    .. math:: \hat{G}_k(x; c, C)
        = \hat{x}^k_C \text{exp}(-c \hat{x}^2_C),

    where :math:`k > 0` is the "angular momentum", :math:`x_C \equiv x - C`
    with :math:`C \in \mathbb{R}` being the center of the Gaussian well, and
    :math:`c > 0 \in \mathbb{R}` the inverse spread of the well.

    Parameters
    ----------
    londons : list[G1D]
        The Gaussian orbitals to use for the matrix elements.
    c : float
        Coefficient in the Gaussian exponential with :math:`c > 0`.
    center : float
        Center of the Gaussian well. Default is in `0.0`, i.e., the origin.
    k : int
        The "angular momentum" of the well, i.e., the exponent of the monomial
        in front of the exponential terms. Default is `0`, i.e., assuming an
        :math:`s`-Gaussian shape of the well.

    Returns
    -------
    np.ndarray
        A real :math:`n \times n` array where :math:`n` is the number of
        Gaussian orbitals passed in, i.e., `n == len(gaussians)`.
    """

    assert c > 0

    op = G1D(k, c, center)

    return _construct_gaussian_operator_matrix_elements(op, gaussians)


def _construct_gaussian_operator_matrix_elements(
    op: G1D, gaussians: list
) -> np.ndarray:
    l = len(gaussians)
    gop_k = np.zeros((l, l))

    for i in range(l):
        g_i = gaussians[i]
        gop_k[i, i] = g_i.norm**2 * G(op, g_i, g_i)

        for j in range(i + 1, l):
            g_j = gaussians[j]
            val = g_i.norm * g_j.norm * G(op, g_i, g_j)

            gop_k[i, j] = val
            gop_k[j, i] = val

    return gop_k


def G(op: G1D, g_i: G1D, g_j: G1D) -> float:
    val = 0
    od = OD1D(g_i, g_j)
    g_0 = G1D(0, od.p, od.P)

    for t in range(od.i + od.j + 1):
        val += od.E(t) * P(t, op, g_0)

    return val


def P(t: int, op: G1D, g_l: G1D) -> float:
    if t == 0:
        od = OD1D(op, g_l)
        return od.E(0) * np.sqrt(np.pi / od.p)

    return 2 * g_l.a * P(t - 1, op, g_l.increment_i()) - (
        0 if g_l.i == 0 else g_l.i * P(t - 1, op, g_l.decrement_i())
    )
