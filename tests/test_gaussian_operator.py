import numpy as np

from gaussians import G1D
from gaussians.one_dim import construct_gaussian_operator_matrix_elements


def test_simplest_case():
    a = 1
    b = 2
    c = 0.5

    gaussians = [G1D(0, a), G1D(0, b)]
    well = G1D(0, c)

    g_e = construct_gaussian_operator_matrix_elements(well, gaussians)
    g_test = np.zeros_like(g_e)

    for i, g_i in enumerate(gaussians):
        for j, g_j in enumerate(gaussians):
            g_test[i, j] = (
                -g_i.norm * g_j.norm * np.sqrt(np.pi / (g_i.a + g_j.a + well.a))
            )

    np.testing.assert_allclose(g_e, g_test)


@__import__("pytest").mark.skip()
def test_origin_center():
    gaussians = [G1D(0, 1), G1D(1, 1.5), G1D(2, 0.7), G1D(0, 0.5)]
    wells = [G1D(0, 0.3), G1D(1, 1.2), G1D(2, 1.1)]
    gaussians = [G1D(2, 1)]
    wells = [G1D(0, 0.3)]
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
        g_e = construct_gaussian_operator_matrix_elements(well, gaussians)
        g_test = np.zeros_like(g_e)

        for i, g_i in enumerate(gaussians):
            for j, g_j in enumerate(gaussians):
                ang_sum = g_i.i + g_j.i + well.i
                exp_sum = g_i.a + g_j.a + well.a

                g_test[i, j] = g_i.norm * g_j.norm * res_func[ang_sum](exp_sum)
                print(i, j, g_i.i, g_j.i, well.i, ang_sum, exp_sum)
                # print(res_func[ang_sum](1))
                # np.testing.assert_allclose(g_e[i, j], g_test[i, j])

        # print(well.i)
        np.testing.assert_allclose(g_e, g_test)
        np.testing.assert_allclose(g_e, g_e.T)
        wat
