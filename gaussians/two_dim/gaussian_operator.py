import numpy as np

from .g2d import G2D
from gaussians.one_dim import (
    G1D,
    construct_gaussian_operator_matrix_elements as go_1d,
)


def construct_gaussian_operator_matrix_elements(
    gaussians: list,
    c: (float, float),
    center=(0.0, 0.0),
    k=(0, 0),
) -> np.ndarray:
    x_gaussians = list()
    y_gaussians = list()

    for g in gaussians:
        x_gaussians.append(g.G_x)
        y_gaussians.append(g.G_y)

    return go_1d(x_gaussians, c[0], center=center[0], k=k[0]) * go_1d(
        y_gaussians, c[1], center=center[1], k=k[1]
    )
