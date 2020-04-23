import numba
import numpy as np

from .g1d import G1D
from .od1d import OD1D


def construct_multipole_moment_matrix(
    e: int, C: float, gaussians: list
) -> np.ndarray:
    r"""Function constructing all multipole moment matrix elements for a
    specific order ``e``, center ``C``, and a given list of Gaussians.

    >>> from gaussians.one_dim.g1d import G1D
    >>> s = construct_multipole_moment_matrix(0, 0, [G1D(0, 2, -4), G1D(0, 2, 4)])
    >>> s.shape
    (2, 2)
    >>> abs(s[0, 0] - s[1, 1]) < 1e-12
    True
    >>> abs(s[0, 1] - s[1, 0]) < 1e-12
    True
    >>> abs(s[0, 1]) < 1e-12
    True
    >>> abs(s[0, 0] - 1) < 1e-12
    True
    >>> abs(s[1, 1] - 1) < 1e-12
    True
    """

    l = len(gaussians)
    s_e = np.zeros((l, l))

    for i in range(l):
        G_i = gaussians[i]
        for j in range(i + 1, l):
            G_j = gaussians[j]

            val = S(e, C, G_i, G_j)
            s_e[i, j] = val
            s_e[j, i] = val

    return s_e


def S(e: int, C: float, G_i: G1D, G_j: G1D) -> float:
    r"""Function computing the one-dimensional multipole moment integral

    .. math:: S^{e}_{ij} = \langle G_i \rvert \hat{x}^{e}_C
        \lvert G_j \rangle,

    where :math:`G_i(x; a, A)` is a one-dimensional primitive Gaussian with
    angular momentum :math:`i`, spread :math:`a`, center at :math:`A`, and
    the multipole is centered around :math:`C` with :math:`x_c = x - C`.

    >>> import numpy as np
    >>> from gaussians import G1D
    >>> G_0 = G1D(0, 1, 0)
    >>> abs(S(0, 0, G_0, G_0) - 1) < 1e-14
    True
    >>> G_1 = G1D(1, 1, 0)
    >>> abs(S(0, 0, G_1, G_1) - 1) < 1e-14
    True
    >>> G_4 = G1D(4, 2, 1)
    >>> abs(S(0, 0, G_4, G_4) - 1) < 1e-14
    True
    """

    return S_od(e, C, OD1D(G_i, G_j))


def S_od(e: int, C: float, O_ij: OD1D) -> float:
    r"""Function computing the one-dimensional multipole moment integral, but
    this function skips the construction of the overlap distribution as there
    might be something to gain from precomputing all the overlap distributions
    in advance.

    >>> import numpy as np
    >>> from gaussians import G1D, OD1D
    >>> G_4 = G1D(4, 1, 0)
    >>> G_3 = G1D(3, 0.3, 1)
    >>> O_43 = OD1D(G_4, G_3)
    >>> C = 0.5
    >>> S_1_43 = S_od(1, C, O_43)
    >>> abs(
    ...     S_1_43
    ...     - (
    ...         O_43.norm
    ...         * (
    ...             O_43.E(1)
    ...             + (O_43.P - C) * O_43.E(0)
    ...         ) * np.sqrt(np.pi / O_43.p)
    ...     )
    ... ) < 1e-14
    True
    """

    val = 0

    for t in range(min(O_ij.i + O_ij.j, e) + 1):
        val += O_ij.E(t) * M(e, t, O_ij.p, O_ij.P, C)

    return val * O_ij.norm


@numba.njit(cache=True, fastmath=True, nogil=True)
def M(e: int, t: int, p: float, P: float, C: float) -> float:
    r"""Function computing Hermite multipole moment integrals. That is,

    .. math:: M^{e}_{t} = \int^{\infty}_{-\infty}
        x^{e}_{C} \Lambda_{t} \mathrm{d}x,

    where :math:`x_C \equiv x - C` with :math:`C` being a constant.
    """

    if t > e:
        return 0

    if t < 0 or e < 0:
        return 0

    if e == 0:
        return (t == 0) * np.sqrt(np.pi / p)

    X_PC = P - C

    return (
        t * M(e - 1, t - 1, p, P, C)
        + X_PC * M(e - 1, t, p, P, C)
        + 1 / (2 * p) * M(e - 1, t + 1, p, P, C)
    )
