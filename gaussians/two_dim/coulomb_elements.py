import numpy as np
import scipy.special

from .g2d import G2D
from .od2d import OD2D


def construct_coulomb_interaction_matrix_elements(gaussians: list):
    l = len(gaussians)

    u = np.zeros((l, l, l, l))

    for a, G_a in enumerate(gaussians):
        for b, G_b in enumerate(gaussians):
            for c, G_c in enumerate(gaussians):
                for d, G_d in enumerate(gaussians):
                    u[a, b, c, d] = (
                        G_a.norm
                        * G_b.norm
                        * G_c.norm
                        * G_d.norm
                        * construct_coulomb_interaction_matrix_element(
                            G_a, G_b, G_c, G_d
                        )
                    )

    return u


def construct_coulomb_interaction_matrix_element(
    G_a: G2D, G_b: G2D, G_c: G2D, G_d: G2D
) -> float:
    O_ac = OD2D(G_a, G_c)
    O_bd = OD2D(G_b, G_d)

    p = O_ac.p
    q = O_bd.p
    P = O_ac.P
    Q = O_bd.P

    sigma = (p + q) / (4 * p * q)
    arg = 1 / (4 * sigma)
    delta = Q - P

    val = 0

    for t in range(O_ac.x_sum_lim + 1):
        for u in range(O_ac.y_sum_lim + 1):
            # E_ac = O_ac.E(t, u)
            E_ac = (-1) ** (t + u) * O_ac.E(t, u)
            for tau in range(O_bd.x_sum_lim + 1):
                for nu in range(O_bd.y_sum_lim + 1):
                    # E_bd = (-1) ** (tau + nu) * O_bd.E(tau, nu)
                    E_bd = O_bd.E(tau, nu)

                    val += E_ac * E_bd * I_twiddle(t + tau, u + nu, arg, delta)

    return np.pi ** 2 / (p * q) * np.sqrt(np.pi / (4 * sigma)) * val


def I_twiddle(t: int, u: int, p: float, sigma: np.ndarray) -> float:
    return _I_twiddle(0, t, u, p, sigma)


def _I_twiddle(n: int, t: int, u: int, p: float, sigma: np.ndarray) -> float:
    assert n >= -1
    assert t >= -1
    assert u >= -1

    if t < 0 or u < 0:
        return 0

    if n == -1:
        return _I_twiddle(-n, t, u, p, sigma)

    if t == u == 0:
        arg = -p * np.sum(sigma ** 2) / 2
        return scipy.special.ive(n, arg)

    pre_factor = -p / 2

    if t > 0:
        return pre_factor * (
            (t - 1)
            * (
                _I_twiddle(n - 1, t - 2, u, p, sigma)
                + 2 * _I_twiddle(n, t - 2, u, p, sigma)
                + _I_twiddle(n + 1, t - 2, u, p, sigma)
            )
            + sigma[0]
            * (
                _I_twiddle(n - 1, t - 1, u, p, sigma)
                + 2 * _I_twiddle(n, t - 1, u, p, sigma)
                + _I_twiddle(n + 1, t - 1, u, p, sigma)
            )
        )

    return pre_factor * (
        (u - 1)
        * (
            _I_twiddle(n - 1, t, u - 2, p, sigma)
            + 2 * _I_twiddle(n, t, u - 2, p, sigma)
            + _I_twiddle(n + 1, t, u - 2, p, sigma)
        )
        + sigma[1]
        * (
            _I_twiddle(n - 1, t, u - 1, p, sigma)
            + 2 * _I_twiddle(n, t, u - 1, p, sigma)
            + _I_twiddle(n + 1, t, u - 1, p, sigma)
        )
    )
