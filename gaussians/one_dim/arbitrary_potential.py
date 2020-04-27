import numpy as np
import scipy.integrate


def construct_arbitrary_potential_elements(
    potential: callable, gaussians: list, grid: np.ndarray
) -> np.ndarray:
    l = len(gaussians)
    pot_grid = potential(grid)

    v = np.zeros((l, l), dtype=pot_grid.dtype)

    for i in range(l):
        G_i_grid = gaussians[i](grid, with_norm=True)

        val = scipy.integrate.simps(G_i_grid * pot_grid * G_i_grid, grid)

        v[i, i] = val

        for j in range(i + 1, l):
            G_j_grid = gaussians[j](grid, with_norm=True)

            val = scipy.integrate.simps(G_i_grid * pot_grid * G_j_grid, grid)
            v[i, j] = val
            v[j, i] = np.conjugate(val)

    return v
