import numpy as np

from scipy.special import hyp1f1


def boys(n: int, x: float) -> float:
    # TODO: This function should be jitted. Test numba-scipy solution found here:
    # https://github.com/numba/numba-scipy
    return hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1)


def R(
    n: int,
    t: int,
    u: int,
    v: int,
    a: float,
    A: np.ndarray,
    b: float,
    B: np.ndarray,
    C: np.ndarray,
) -> float:
    pass
