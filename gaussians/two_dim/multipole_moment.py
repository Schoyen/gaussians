import numpy as np

from gaussians.one_dim import (
    construct_multipole_moment_matrix_elements as mm_1d,
)

# from gaussians.one_dim.multipole_moment import S


def construct_multipole_moment_matrix_elements(
    e: [int, int], C: [float, float], gaussians: list
) -> np.ndarray:

    x_gaussians = list()
    y_gaussians = list()

    for G in gaussians:
        x_gaussians.append(G.G_x)
        y_gaussians.append(G.G_y)

    return mm_1d(e[0], C[0], x_gaussians) * mm_1d(e[1], C[1], y_gaussians)

    # l = len(gaussians)

    # s_ef = np.zeros((l, l))

    # for i in range(l):
    #     G_i = gaussians[i]

    #     s_ef[i, i] = (
    #         G_i.norm ** 2
    #         * S(e[0], C[0], G_i.G_x, G_i.G_x)
    #         * S(e[1], C[1], G_i.G_y, G_i.G_y)
    #     )

    #     for j in range(i + 1, l):
    #         G_j = gaussians[j]

    #         val = (
    #             G_i.norm
    #             * G_j.norm
    #             * S(e[0], C[0], G_i.G_x, G_j.G_x)
    #             * S(e[1], C[1], G_i.G_y, G_j.G_y)
    #         )

    #         s_ef[i, j] = val
    #         s_ef[j, i] = val

    # return s_ef
