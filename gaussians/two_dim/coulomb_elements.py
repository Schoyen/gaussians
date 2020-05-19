import numba
import numpy as np
import scipy.special
import numba_scipy

from .od2d import OD2D


def construct_coulomb_matrix_elements(gaussians: list):
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
                        * construct_coulomb_matrix_element(G_a, G_b, G_c, G_d)
                    )

    return u


def construct_coulomb_matrix_element(G_a, G_b, G_c, G_d):
    O_ac = OD2D(G_a, G_c)
    O_bd = OD2D(G_b, G_d)

    p = O_ac.p
    q = O_bd.p
    P = O_ac.P
    Q = O_bd.P

    sigma = (p + q) / (4 * p * q)
    delta = Q - P

    val = 0

    for t in range(O_ac.x_sum_lim + 1):
        for u in range(O_ac.y_sum_lim + 1):
            E_ac = O_ac.E(t, u)
            # E_ac = (-1) ** (t + u) * O_ac.E(t, u)
            for tau in range(O_bd.x_sum_lim + 1):
                for nu in range(O_bd.y_sum_lim + 1):
                    E_bd = (-1) ** (tau + nu) * O_bd.E(tau, nu)
                    # E_bd = O_bd.E(tau, nu)

                    val += E_ac * E_bd * I_tilde(t + tau, u + nu, sigma, delta)

    return np.pi ** 2 / (p * q) * np.sqrt(np.pi / (4 * sigma)) * val


# @numba.njit(cache=True)
def I_tilde(t, u, sigma, delta):
    return _I_tilde(0, t, u, sigma, delta)


# @numba.njit(cache=True)
def _I_tilde(n, t, u, sigma, delta):
    assert n >= 0

    if t < 0 or u < 0:
        return 0

    if t == u == 0:
        return extended_bessel(n, sigma, delta)

    pre_factor = 1 / (8 * sigma)

    val = 0

    if n == 0:
        pre_factor *= 2

        if t == 0:
            val += delta[1] * (
                _I_tilde(n, t, u - 1, sigma, delta)
                + _I_tilde(n + 1, t, u - 1, sigma, delta)
            )

            if u > 1:
                val += -(u - 1) * (
                    _I_tilde(n, t, u - 2, sigma, delta)
                    + _I_tilde(n + 1, t, u - 2, sigma, delta)
                )

            return val * pre_factor

        val += delta[0] * (
            _I_tilde(n, t - 1, u, sigma, delta)
            + _I_tilde(n + 1, t - 1, u, sigma, delta)
        )

        if t > 1:
            val += -(t - 1) * (
                _I_tilde(n, t - 2, u, sigma, delta)
                + _I_tilde(n + 1, t - 2, u, sigma, delta)
            )

        return val * pre_factor

    if t == 0:
        val += delta[1] * (
            _I_tilde(n - 1, t, u - 1, sigma, delta)
            + 2 * _I_tilde(n, t, u - 1, sigma, delta)
            + _I_tilde(n + 1, t, u - 1, sigma, delta)
        )

        if u > 1:
            val += -(u - 1) * (
                _I_tilde(n - 1, t, u - 2, sigma, delta)
                + 2 * _I_tilde(n, t, u - 2, sigma, delta)
                + _I_tilde(n + 1, t, u - 2, sigma, delta)
            )

        return val * pre_factor

    val += delta[0] * (
        _I_tilde(n - 1, t - 1, u, sigma, delta)
        + 2 * _I_tilde(n, t - 1, u, sigma, delta)
        + _I_tilde(n + 1, t - 1, u, sigma, delta)
    )

    if t > 1:
        val += -(t - 1) * (
            _I_tilde(n - 1, t - 2, u, sigma, delta)
            + 2 * _I_tilde(n, t - 2, u, sigma, delta)
            + _I_tilde(n + 1, t - 2, u, sigma, delta)
        )

    return val * pre_factor

    # if u == 0:
    #     return pre_factor * (
    #         delta[0]
    #         * (
    #             _I_tilde(n_1, t - 1, u, sigma, delta)
    #             + _I_tilde(n + 1, t - 1, u, sigma, delta)
    #         )
    #         - t
    #         * (
    #             _I_tilde(n_1, t - 2, u, sigma, delta)
    #             + _I_tilde(n + 1, t - 2, u, sigma, delta)
    #         )
    #     )

    # return pre_factor * (
    #     delta[1]
    #     * (
    #         _I_tilde(n_1, t, u - 1, sigma, delta)
    #         + _I_tilde(n + 1, t, u - 1, sigma, delta)
    #     )
    #     - u
    #     * (
    #         _I_tilde(n_1, t, u - 2, sigma, delta)
    #         + _I_tilde(n + 1, t, u - 2, sigma, delta)
    #     )
    # )


# @numba.njit(cache=True)
def extended_bessel(n, sigma, delta):
    delta_sq = delta[0] ** 2 + delta[1] ** 2
    arg = -delta_sq / (8 * sigma)

    return np.exp(arg) * scipy.special.iv(n, arg)
