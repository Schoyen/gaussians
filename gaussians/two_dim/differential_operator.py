import numpy as np

from gaussians.one_dim import construct_differential_matrix_elements as dm_1d


def construct_differential_matrix_elements(
    e: [int, int], gaussians: list
) -> np.ndarray:
    x_gaussians = list()
    y_gaussians = list()

    for G in gaussians:
        x_gaussians.append(G.G_x)
        y_gaussians.append(G.G_y)

    return dm_1d(e[0], x_gaussians) * dm_1d(e[1], x_gaussians)
