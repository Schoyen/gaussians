import pytest

import numpy as np


@pytest.mark.skip
def test_mut_arr():
    from gaussians.gaussian_lib import mul_arr

    a = np.random.random((100, 57))
    b = a.copy()

    mul_arr(a)

    np.testing.assert_allclose(2 * b, a)
