import numpy as np

from gaussians import G1D
from gaussians.one_dim import construct_diff_mm_matrix_elements
import gaussians.one_dim_lib as odl


def test_comparison_diff_mm():
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

        l_ef = construct_diff_mm_matrix_elements(e, f, center, gaussians)
        l_ef_r = odl.construct_diff_mm_matrix_elements(
            e, f, center, [g.get_params() for g in gaussians]
        )

        np.testing.assert_allclose(l_ef, l_ef_r)
