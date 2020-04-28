import numpy as np

from gaussians.one_dim import construct_differential_matrix_elements as dm_1d


def construct_kinetic_matrix_elements(gaussians: list):
    return -0.5 * (
        construct_differential_matrix_elements([2, 0], gaussians)
        + construct_differential_matrix_elements([0, 2], gaussians)
    )


def construct_differential_matrix_elements(
    e: [int, int], gaussians: list
) -> np.ndarray:
    x_gaussians = list()
    y_gaussians = list()

    for G in gaussians:
        x_gaussians.append(G.G_x)
        y_gaussians.append(G.G_y)

    return dm_1d(e[0], x_gaussians) * dm_1d(e[1], y_gaussians)
