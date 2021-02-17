import numpy as np


def test_mut_arr():
    from gaussians.gaussian_lib import mul_arr

    a = np.random.random((5, 4))
    b = a.copy()

    mul_arr(a)

    np.testing.assert_allclose(2 * b, a)
