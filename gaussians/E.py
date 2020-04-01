import numba
import numpy as np


@numba.njit(cache=True, fastmath=True, nogil=True)
def E(i: int, j: int, t: int, a: float, A: float, b: float, B: float) -> float:
    r"""Function computing coefficients transforming Hermite-Gaussians to an
    overlap distribution function between two Cartesian Primitive Gaussians.
    """
    if t < 0 or t > i + j:
        return 0

    if i < 0 or j < 0:
        return 0

    mu = a * b / (a + b)
    X_AB = A - B

    if t == 0:
        return np.exp(-mu * X_AB ** 2)

    p = a + b
    P = (a * A + b * B) / p

    if i == 0:
        X_PB = P - B

        return (
            1 / (2 * p) * E(i, j - 1, t - 1, a, A, b, B)
            + X_PB * E(i, j - 1, t, a, A, b, B)
            + (t + 1) * E(i, j - 1, t + 1, a, A, b, B)
        )

    X_PA = P - A

    return (
        1 / (2 * p) * E(i - 1, j, t - 1, a, A, b, B)
        + X_PA * E(i - 1, j, t, a, A, b, B)
        + (t + 1) * E(i - 1, j, t + 1, a, A, b, B)
    )
