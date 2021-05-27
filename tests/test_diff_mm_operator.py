import numpy as np

from gaussians import G1D, G2D
from gaussians.one_dim import construct_diff_mm_matrix_elements as dm_1d
from gaussians.two_dim import construct_diff_mm_matrix_elements as dm_2d
import gaussians.one_dim_lib as odl
import gaussians.two_dim_lib as tdl


def test_comparison_dm_1d():
    for i in range(3):
        gaussians = [
            G1D(
                np.random.choice(range(4)),
                np.random.random() + 0.001,
                2 * (np.random.random() - 0.5),
            )
            for i in range(5)
        ]

        e = np.random.choice(range(4))
        f = np.random.choice(range(4))
        center = 2 * (np.random.random() - 0.5)

        l_ef = dm_1d(e, f, center, gaussians)
        l_ef_r = odl.construct_diff_mm_matrix_elements(
            e, f, center, [g.get_params() for g in gaussians]
        )

        np.testing.assert_allclose(l_ef, l_ef_r)


def test_ang_mom_z():
    for i in range(3):
        gaussians = [
            G2D(
                [np.random.choice(range(4)), np.random.choice(range(4))],
                np.random.random() + 0.001,
                [
                    2 * (np.random.random() - 0.5),
                    2 * (np.random.random() - 0.5),
                ],
            )
            for i in range(5)
        ]

        l_z = dm_2d([1, 0], [0, 1], [0, 0], gaussians) - dm_2d(
            [0, 1], [1, 0], [0, 0], gaussians
        )
        l_z_r = tdl.construct_angular_moment_z_matrix_elements(
            [g.get_params() for g in gaussians]
        )

        np.testing.assert_allclose(l_z, l_z_r)
