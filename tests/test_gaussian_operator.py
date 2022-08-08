import numpy as np

from gaussians import G1D, G2D
from gaussians.one_dim.gaussian_operator import G
from gaussians.one_dim import (
    construct_gaussian_operator_matrix_elements as go_1d,
)
from gaussians.two_dim import (
    construct_gaussian_operator_matrix_elements as go_2d,
)
import gaussians.one_dim_lib as odl
import gaussians.two_dim_lib as tdl


def test_simplest_case_1d():
    a = np.random.random() + 0.1
    b = np.random.random() + 0.1
    c = np.random.random() + 0.1

    gaussians = [G1D(0, a), G1D(0, b)]
    well = G1D(0, c)

    g_e = -go_1d(gaussians, c)
    g_e_r = -odl.construct_gaussian_operator_matrix_elements(
        [well.get_params()], [g.get_params() for g in gaussians]
    )
    g_test = np.zeros_like(g_e)

    for i, g_i in enumerate(gaussians):
        for j, g_j in enumerate(gaussians):
            g_test[i, j] = (
                -g_i.norm * g_j.norm * np.sqrt(np.pi / (g_i.a + g_j.a + well.a))
            )

    np.testing.assert_allclose(g_e, g_test)
    np.testing.assert_allclose(g_e, g_e_r)


def test_simplest_case_2d():
    a = np.random.random() + 0.1
    b = np.random.random() + 0.1
    c = np.random.random() + 0.1

    gaussians = [G2D((0, 0), a), G2D((0, 0), b)]
    well = G2D((0, 0), c)

    g_e = -go_2d(gaussians, (c, c))
    g_e_r = -tdl.construct_gaussian_operator_matrix_elements(
        [well.get_params()], [g.get_params() for g in gaussians]
    )

    g_test = np.zeros_like(g_e)

    for i, g_i in enumerate(gaussians):
        for j, g_j in enumerate(gaussians):
            g_test[i, j] = (
                -g_i.norm * g_j.norm * np.pi / (g_i.a + g_j.a + well.a)
            )

    np.testing.assert_allclose(g_e, g_test)
    np.testing.assert_allclose(g_e, g_e_r)


def test_symmetry_of_integrals():
    center = np.random.random()
    gaussians = [
        G1D(0, 1, A=center),
        G1D(1, 1.5, A=center),
        G1D(2, 0.7, A=center),
        G1D(0, 0.5, A=center),
    ]
    wells = [
        G1D(0, 0.3, A=center),
        G1D(1, 1.2, A=center),
        G1D(2, 1.1, A=center),
    ]

    for well in wells:

        for i, g_i in enumerate(gaussians):
            for j, g_j in enumerate(gaussians):
                val = g_i.norm * g_j.norm * G(well, g_i, g_j)
                val_2 = g_i.norm * g_j.norm * G(well, g_j, g_i)
                val_3 = g_i.norm * g_j.norm * G(g_i, g_j, well)
                val_4 = g_i.norm * g_j.norm * G(g_j, g_i, well)
                val_5 = g_i.norm * g_j.norm * G(g_j, well, g_i)
                val_6 = g_i.norm * g_j.norm * G(g_i, well, g_j)

                assert abs(val - val_2) < 1e-12
                assert abs(val - val_3) < 1e-12
                assert abs(val - val_4) < 1e-12
                assert abs(val - val_5) < 1e-12
                assert abs(val - val_6) < 1e-12


def test_common_center_1d():
    center = np.random.random()
    gaussians = [
        G1D(0, 1, A=center),
        G1D(1, 1.5, A=center),
        G1D(2, 0.7, A=center),
        G1D(0, 0.5, A=center),
    ]
    wells = [
        G1D(0, 0.3, A=center),
        G1D(1, 1.2, A=center),
        G1D(2, 1.1, A=center),
    ]
    res_func = [
        lambda a: -np.sqrt(np.pi / a),
        lambda a: 0,
        lambda a: -np.sqrt(np.pi) / (2 * a ** (3 / 2)),
        lambda a: 0,
        lambda a: -(3 / 4) * np.sqrt(np.pi) / (a ** (5 / 2)),
        lambda a: 0,
        lambda a: -(15 / 8) * np.sqrt(np.pi) / (a ** (7 / 2)),
    ]

    for well in wells:
        g_e = -go_1d(gaussians, well.a, center=well.A, k=well.i)
        g_e_r = -odl.construct_gaussian_operator_matrix_elements(
            [well.get_params()], [g.get_params() for g in gaussians]
        )
        g_test = np.zeros_like(g_e)

        for i, g_i in enumerate(gaussians):
            for j, g_j in enumerate(gaussians):
                ang_sum = g_i.i + g_j.i + well.i
                exp_sum = g_i.a + g_j.a + well.a

                g_test[i, j] = g_i.norm * g_j.norm * res_func[ang_sum](exp_sum)

        np.testing.assert_allclose(g_e, g_test, atol=1e-12)
        np.testing.assert_allclose(g_e, g_e.T)
        np.testing.assert_allclose(g_e, g_e_r)


def test_common_center_2d():
    center = [np.random.random(), np.random.random()]
    gaussians = [
        G2D([0, 0], 1, A=center),
        G2D([1, 0], 1.5, A=center),
        G2D([0, 1], 1.3, A=center),
        G2D([1, 1], 2.0, A=center),
        G2D([2, 0], 0.7, A=center),
        G2D([0, 2], 0.7, A=center),
    ]
    wells = [
        G2D([0, 0], 0.3, A=center),
        G2D([1, 0], 1.2, A=center),
        G2D([0, 1], 1.1, A=center),
        G2D([1, 1], 1.3, A=center),
        G2D([2, 0], 1.6, A=center),
        G2D([0, 2], 1.4, A=center),
    ]
    res_func = [
        lambda a: np.sqrt(np.pi / a),
        lambda a: 0,
        lambda a: np.sqrt(np.pi) / (2 * a ** (3 / 2)),
        lambda a: 0,
        lambda a: (3 / 4) * np.sqrt(np.pi) / (a ** (5 / 2)),
        lambda a: 0,
        lambda a: (15 / 8) * np.sqrt(np.pi) / (a ** (7 / 2)),
    ]

    for well in wells:
        g_e = -go_2d(gaussians, c=(well.a, well.a), center=well.A, k=well.alpha)
        g_e_r = -tdl.construct_gaussian_operator_matrix_elements(
            [well.get_params()], [g.get_params() for g in gaussians]
        )
        g_test = np.zeros_like(g_e)

        for i, g_i in enumerate(gaussians):
            for j, g_j in enumerate(gaussians):
                ang_sum_x = g_i.G_x.i + g_j.G_x.i + well.G_x.i
                ang_sum_y = g_i.G_y.i + g_j.G_y.i + well.G_y.i
                exp_sum = g_i.a + g_j.a + well.a

                g_test[i, j] = (
                    -g_i.norm
                    * g_j.norm
                    * res_func[ang_sum_x](exp_sum)
                    * res_func[ang_sum_y](exp_sum)
                )

        np.testing.assert_allclose(g_e, g_test, atol=1e-12)
        np.testing.assert_allclose(g_e, g_e.T)
        np.testing.assert_allclose(g_e, g_e_r)
