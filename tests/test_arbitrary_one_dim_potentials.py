import numpy as np

from gaussians import G1D
from gaussians.one_dim import (
    construct_arbitrary_potential_elements,
    construct_multipole_moment_matrix_elements,
)


class HOPotential:
    def __init__(self, omega):
        self.omega = omega

    def __call__(self, x):
        return 0.5 * self.omega**2 * x**2


class DWPotentialSmooth:
    """
    This is the double-well potential used by J. Kryvi and S. Bøe in their
    thesis work. See Eq. [13.11] in Bøe: https://www.duo.uio.no/handle/10852/37170
    """

    def __init__(self, a=4):
        self.a = a

    def __call__(self, x):
        return (
            (1.0 / (2 * self.a**2))
            * (x + 0.5 * self.a) ** 2
            * (x - 0.5 * self.a) ** 2
        )


def test_ho_potential():
    l = 10
    omega = 0.5
    grid = np.linspace(-10, 10, 1001)

    gaussians = [G1D(i, omega / 2, 0) for i in range(l)]

    v_mm = (
        0.5
        * omega**2
        * construct_multipole_moment_matrix_elements(2, 0, gaussians)
    )
    v_num = construct_arbitrary_potential_elements(
        HOPotential(omega), gaussians, grid
    )

    np.testing.assert_allclose(v_mm, v_num, atol=1e-12)
    np.testing.assert_allclose(v_mm, v_mm.T)


def test_dw_smooth():
    l = 4
    a = 4
    grid = np.linspace(-10, 10, 1001)

    gaussians = [G1D(i, 1, -2) for i in range(l // 2)]
    gaussians.extend([G1D(i, 1, 2) for i in range(l // 2)])

    s = construct_multipole_moment_matrix_elements(0, 0, gaussians)
    x_2 = construct_multipole_moment_matrix_elements(2, 0, gaussians)
    x_4 = construct_multipole_moment_matrix_elements(4, 0, gaussians)

    v_mm = 1 / (2 * a**2) * (x_4 - 0.5 * a**2 * x_2 + 1 / 16 * a**4 * s)
    v_num = construct_arbitrary_potential_elements(
        DWPotentialSmooth(a), gaussians, grid
    )

    np.testing.assert_allclose(v_mm, v_num, atol=1e-12)
    np.testing.assert_allclose(v_mm, v_mm.T)
