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
    if n < 0 or t < 0 or u < 0 or v < 0:
        return 0

    p = a + b
    P = (a * A + b * B) / p

    PC = P - C
    R_PC = np.linalg.norm(PC)

    if t == u == v == 0:
        return (-2 * p) ** n * boys(n, p * R_PC**2)

    if t == u == 0:
        return (v - 1) * R(n + 1, t, u, v - 2, a, A, b, B, C) + PC[2] * R(
            n + 1, t, u, v - 1, a, A, b, B, C
        )
    elif t == 0:
        return (u - 1) * R(n + 1, t, u - 2, v, a, A, b, B, C) + PC[1] * R(
            n + 1, t, u - 1, v, a, A, b, B, C
        )

    return (t - 1) * R(n + 1, t - 2, u, v, a, A, b, B, C) + PC[0] * R(
        n + 1, t - 1, u, v, a, A, b, B, C
    )


if __name__ == "__main__":
    print(
        R(
            3,
            2,
            2,
            2,
            1,
            np.array([0, 0, 0]),
            1,
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )
    )
