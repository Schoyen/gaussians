import numpy as np

from gaussians import G2D
from gaussians.two_dim import (
    construct_multipole_moment_matrix_elements,
    construct_arbitrary_potential_elements,
    construct_arbitrary_separable_potential_elements,
)


def test_dipole_moment():
    x = np.linspace(-6, 6, 201)
    y = np.linspace(-6, 6, 201)

    potential = lambda x: x
    overlap = lambda x: np.ones(x.shape)

    omega = 1
    l = 6

    shell = l // 2 + l % 2

    gaussians = []

    for i in range(shell):
        for j in range(i, shell - i):
            gaussians.append(
                G2D(
                    (i, j),
                    omega / 2,
                    A=[np.random.random(), np.random.random()],
                )
            )

            if i != j:
                gaussians.append(G2D((j, i), omega / 2))

    x_an = construct_multipole_moment_matrix_elements([1, 0], [0, 0], gaussians)
    y_an = construct_multipole_moment_matrix_elements([0, 1], [0, 0], gaussians)

    x_sep = construct_arbitrary_separable_potential_elements(
        potential, overlap, gaussians, x, y,
    )

    y_sep = construct_arbitrary_separable_potential_elements(
        overlap, potential, gaussians, x, y,
    )

    np.testing.assert_allclose(x_an, x_sep, atol=1e-10)
    np.testing.assert_allclose(y_an, y_sep, atol=1e-10)

    x_full = construct_arbitrary_potential_elements(
        lambda x, y: x, gaussians, x, y
    )
    y_full = construct_arbitrary_potential_elements(
        lambda x, y: y, gaussians, x, y
    )

    np.testing.assert_allclose(x_an, x_full, atol=1e-10)
    np.testing.assert_allclose(y_an, y_full, atol=1e-10)
