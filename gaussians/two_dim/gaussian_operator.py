import numpy as np

from .g2d import G2D
from gaussians.one_dim import (
    construct_gaussian_operator_matrix_elements as go_1d,
)


def construct_gaussian_operator_matrix_elements(
    op: G2D, gaussians: list
) -> np.ndarray:
    x_gaussians = list()
    y_gaussians = list()

    for g in gaussians:
        x_gaussians.append(g.G_x)
        y_gaussians.append(g.G_y)

    return go_1d(op.G_x, x_gaussians) * go_1d(op.G_y, y_gaussians)
