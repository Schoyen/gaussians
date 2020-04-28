import numpy as np

from gaussians.one_dim import (
    construct_multipole_moment_matrix_elements as mm_1d,
)


def construct_overlap_matrix_elements(gaussians: list):
    return construct_multipole_moment_matrix_elements([0, 0], [0, 0], gaussians)


def construct_multipole_moment_matrix_elements(
    e: [int, int], C: [float, float], gaussians: list
) -> np.ndarray:

    x_gaussians = list()
    y_gaussians = list()

    for G in gaussians:
        x_gaussians.append(G.G_x)
        y_gaussians.append(G.G_y)

    return mm_1d(e[0], C[0], x_gaussians) * mm_1d(e[1], C[1], y_gaussians)
