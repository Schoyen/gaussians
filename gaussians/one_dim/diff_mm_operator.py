import numpy as np


from .g1d import G1D
from .od1d import OD1D


def construct_overlap_matrix_elements(gaussians: list) -> np.ndarray:
    return construct_diff_mm_matrix_elements(0, 0, 0, gaussians)


def construct_kinetic_matrix_elements(gaussians: list) -> np.ndarray:
    return -0.5 * construct_diff_mm_matrix_elements(0, 2, 0, gaussians)


def construct_differential_matrix_elements(
    f: int, gaussians: list
) -> np.ndarray:
    return construct_diff_mm_matrix_elements(0, f, 0, gaussians)


def construct_multipole_moment_matrix_elements(
    e: int, C: float, gaussians: list
) -> np.ndarray:
    return construct_diff_mm_matrix_elements(e, 0, C, gaussians)


def construct_diff_mm_matrix_elements(
    e: int, f: int, C: float, gaussians: list
) -> np.ndarray:
    assert e >= 0 and f >= 0

    l = len(gaussians)
    l_ef = np.zeros((l, l))

    for i in range(l):
        g_i = gaussians[i]

        for j in range(l):
            g_j = gaussians[j]
            l_ef[i, j] = g_i.norm * g_j.norm * L(e, f, C, g_i, g_j)

    return l_ef


def L(e: int, f: int, C: float, g_i: G1D, g_j: G1D) -> float:
    if e == 0 and f == 0:
        od_ij = OD1D(g_i, g_j)
        return od_ij.E(0) * np.sqrt(np.pi / od_ij.p)

    if e == 0:
        return g_j.i * (
            L(e, f - 1, C, g_i, g_j.decrement_i()) if g_j.i > 0 else 0
        ) - 2 * g_j.a * L(e, f - 1, C, g_i, g_j.increment_i())

    return L(e - 1, f, C, g_i.increment_i(), g_j) + (g_i.A - C) * L(
        e - 1, f, C, g_i, g_j
    )
