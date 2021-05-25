import numpy as np

from .g1d import G1D
from .od1d import OD1D


def construct_gaussian_operator_matrix_elements(
    op: G1D, gaussians: list
) -> np.ndarray:
    l = len(gaussians)
    gop_k = np.zeros((l, l))

    for i in range(l):
        g_i = gaussians[i]
        gop_k[i, i] = g_i.norm ** 2 * G(op, g_i, g_i)

        for j in range(i + 1, l):
            g_j = gaussians[j]
            val = g_i.norm * g_j.norm * G(op, g_i, g_j)

            gop_k[i, j] = val
            gop_k[j, i] = val

    return gop_k


def G(op: G1D, g_i: G1D, g_j: G1D) -> float:
    val = 0
    od = OD1D(g_i, g_j)
    g_0 = G1D(0, od.p, od.P)

    for t in range(od.i + od.j + 1):
        val += od.E(t) * P(t, op, g_0)

    return -val


def P(t: int, op: G1D, g_l: G1D) -> float:
    if t == 0:
        od = OD1D(op, g_l)
        return od.E(0) * np.sqrt(np.pi / od.p)

    return 2 * g_l.a * P(t - 1, op, g_l.increment_i()) - (
        0 if g_l.i == 0 else g_l.i * P(t - 1, op, g_l.decrement_i())
    )
