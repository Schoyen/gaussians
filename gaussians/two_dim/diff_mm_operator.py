import numpy as np

from gaussians.one_dim import construct_diff_mm_matrix_elements as dm_1d


def construct_overlap_matrix_elements(gaussians: list) -> np.ndarray:
    return construct_diff_mm_matrix_elements([0, 0], [0, 0], [0, 0], gaussians)


def construct_kinetic_matrix_elements(gaussians: list) -> np.ndarray:
    return -0.5 * (
        construct_diff_mm_matrix_elements([0, 0], [2, 0], [0, 0], gaussians)
        + construct_diff_mm_matrix_elements([0, 0], [0, 2], [0, 0], gaussians)
    )


def construct_differential_matrix_elements(
    f: [int, int], gaussians: list
) -> np.ndarray:
    return construct_diff_mm_matrix_elements([0, 0], f, [0, 0], gaussians)


def construct_multipole_moment_matrix_elements(
    e: [int, int], C: [float, float], gaussians: list
) -> np.ndarray:
    return construct_diff_mm_matrix_elements(e, [0, 0], C, gaussians)


def construct_diff_mm_matrix_elements(
    e: [int, int], f: [int, int], C: [float, float], gaussians: list
) -> np.ndarray:
    x_gaussians = list()
    y_gaussians = list()

    for g in gaussians:
        x_gaussians.append(g.G_x)
        y_gaussians.append(g.G_y)

    return dm_1d(e[0], f[0], C[0], x_gaussians) * dm_1d(
        e[1], f[1], C[1], y_gaussians
    )
