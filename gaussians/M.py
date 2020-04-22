import numba
import numpy as np


@numba.njit(cache=True, fastmath=True, nogil=True)
def M(
    e: int, t: int, a: float, A: float, b: float, B: float, C: float
) -> float:
    r"""Function computing Hermite multipole moment integrals. That is,

    .. math:: M^{e}_{t} = \int^{\infty}_{-\infty}
        x^{e}_{C} \Lambda_{t} \mathrm{d}x,

    where :math:`x_C \equiv x - C` with :math:`C` being a constant.
    """

    if t > e:
        return 0

    if t < 0 or e < 0:
        return 0

    p = a + b

    if e == 0:
        return (t == 0) * np.sqrt(np.pi / p)

    P = (a * A + b * B) / p
    X_PC = P - C

    return (
        t * M(e - 1, t - 1, a, A, b, B, C)
        + X_PC * M(e - 1, t, a, A, b, B, C)
        + 1 / (2 * p) * M(e - 1, t + 1, a, A, b, B, C)
    )
