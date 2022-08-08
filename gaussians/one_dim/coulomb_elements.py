import numba
import numpy as np


@numba.njit(cache=True)
def _trapz(f, x):
    n = len(x)
    delta_x = x[1] - x[0]
    val = 0

    for i in range(1, n):
        val += f[i - 1] + f[i]

    return 0.5 * val * delta_x


@numba.njit(cache=True)
def _shielded_coulomb(x_1, x_2, alpha, a):
    return alpha / np.sqrt((x_1 - x_2) ** 2 + a ** 2)


@numba.njit(cache=True)
def _construct_inner_shielded_coulomb_integral(spf, grid, alpha, a):
    num_grid_points = len(grid)
    l = len(spf)
    inner_integral = np.zeros((l, l, num_grid_points))

    for q in range(l):
        for s in range(l):
            for i in range(num_grid_points):
                inner_integral[q, s, i] = _trapz(
                    spf[q]
                    * _shielded_coulomb(grid[i], grid, alpha, a)
                    * spf[s],
                    grid,
                )

    return inner_integral


@numba.njit(cache=True)
def _construct_shielded_coulomb_interaction_matrix_elements(
    spf, grid, alpha, a
):
    # Note: The spf are normalized Gaussians evaluated on the grid.
    l = len(spf)
    inner_integral = _construct_inner_shielded_coulomb_integral(
        spf, grid, alpha, a
    )
    u = np.zeros((l, l, l, l))

    for p in range(l):
        for q in range(l):
            for r in range(l):
                for s in range(l):
                    u[p, q, r, s] = _trapz(
                        spf[p] * inner_integral[q, s] * spf[r],
                        grid,
                    )

    return u


def construct_shielded_coulomb_interaction_matrix_elements(
    gaussians, grid, alpha, a
):
    spf = np.asarray([g(grid, with_norm=True) for g in gaussians])
    return _construct_shielded_coulomb_interaction_matrix_elements(
        spf, grid, alpha, a
    )
