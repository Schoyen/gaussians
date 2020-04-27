import numpy as np
import scipy.integrate


def construct_arbitrary_potential_elements(
    potential: callable, gaussians: list, grid: np.ndarray
) -> np.ndarray:
    l = len(gaussians)
    pot_grid = potential(grid)

    v = np.zeros((l, l), dtype=pot_grid.dtype)

    # TODO: Check symmetry
    for i, G_i in enumerate(gaussians):
        G_i_grid = G_i(grid, with_norm=True)
        for j, G_j in enumerate(gaussians):
            G_j_grid = G_j(grid, with_norm=True)

            v[i, j] = scipy.integrate.simps(
                G_i_grid * pot_grid * G_j_grid, grid
            )

    return v
