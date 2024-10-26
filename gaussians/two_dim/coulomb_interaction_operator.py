import numba

import numpy as np
import scipy.special

import numba_scipy

from .g2d import G2D
from .od2d import OD2D


def construct_coulomb_interaction_matrix_elements(gaussians: list):
    l = len(gaussians)

    u = np.zeros((l, l, l, l))

    for a, g_a in enumerate(gaussians):
        u[a, a, a, a] = (
            g_a.norm**4
            * construct_coulomb_interaction_matrix_element(g_a, g_a, g_a, g_a)
        )

        for b, g_b in enumerate(gaussians):
            if b == a:
                continue

            val = (
                g_a.norm
                * g_b.norm**3
                * construct_coulomb_interaction_matrix_element(
                    g_a, g_b, g_b, g_b
                )
            )

            u[a, b, b, b] = val
            u[b, a, b, b] = val
            u[b, b, a, b] = val
            u[b, b, b, a] = val

            val = (
                g_a.norm**2
                * g_b.norm**2
                * construct_coulomb_interaction_matrix_element(
                    g_a, g_b, g_a, g_b
                )
            )

            u[a, b, a, b] = val
            u[b, a, b, a] = val

            val = (
                g_a.norm**2
                * g_b.norm**2
                * construct_coulomb_interaction_matrix_element(
                    g_a, g_a, g_b, g_b
                )
            )

            u[a, a, b, b] = val
            u[b, b, a, a] = val
            u[a, b, b, a] = val
            u[b, a, a, b] = val

            for c, g_c in enumerate(gaussians):
                if c == b or c == a:
                    continue

                val = (
                    g_a.norm**2
                    * g_b.norm
                    * g_c.norm
                    * construct_coulomb_interaction_matrix_element(
                        g_a, g_b, g_a, g_c
                    )
                )

                u[a, b, a, c] = val
                u[b, a, c, a] = val
                u[a, c, a, b] = val
                u[c, a, b, a] = val

                val = (
                    g_a.norm**2
                    * g_b.norm
                    * g_c.norm
                    * construct_coulomb_interaction_matrix_element(
                        g_a, g_a, g_b, g_c
                    )
                )

                u[a, a, b, c] = val
                u[a, a, c, b] = val
                u[b, c, a, a] = val
                u[c, b, a, a] = val
                u[a, b, c, a] = val
                u[b, a, a, c] = val
                u[c, a, a, b] = val
                u[a, c, b, a] = val

                for d, g_d in enumerate(gaussians):
                    if d == c or d == b or d == a:
                        continue

                    val = (
                        g_a.norm
                        * g_b.norm
                        * g_c.norm
                        * g_d.norm
                        * construct_coulomb_interaction_matrix_element(
                            g_a, g_b, g_c, g_d
                        )
                    )

                    u[a, b, c, d] = val
                    u[b, a, d, c] = val
                    u[c, d, a, b] = val
                    u[d, c, b, a] = val

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

    return np.pi**2 / (p * q) * np.sqrt(np.pi / (4 * sigma)) * val


@numba.njit
def I_twiddle(t: int, u: int, p: float, sigma: np.ndarray) -> float:
    return _I_twiddle(0, t, u, p, sigma)


@numba.njit
def _I_twiddle(n: int, t: int, u: int, p: float, sigma: np.ndarray) -> float:
    assert n >= -1
    assert t >= -1
    assert u >= -1

    if t < 0 or u < 0:
        return 0

    if n == -1:
        return _I_twiddle(-n, t, u, p, sigma)

    if t == u == 0:
        arg = -p * np.sum(sigma**2) / 2
        return scipy.special.ive(float(n), arg)

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
