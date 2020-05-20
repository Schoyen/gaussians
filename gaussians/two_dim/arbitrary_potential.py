import numpy as np
import scipy.integrate


def construct_arbitrary_separable_potential_elements(
    potential_x: callable,
    potential_y: callable,
    gaussians: list,
    x: np.array,
    y: np.array,
) -> np.ndarray:

    l = len(gaussians)

    pot_x = potential_x(x)
    pot_y = potential_y(y)

    v = np.zeros((l, l), dtype=pot_x.dtype)

    for i in range(l):
        G_i_x = gaussians[i].G_x(x, with_norm=True)
        G_i_y = gaussians[i].G_y(y, with_norm=True)

        val_x = scipy.integrate.simps(G_i_x * pot_x * G_i_x, x)
        val_y = scipy.integrate.simps(G_i_y * pot_y * G_i_y, y)

        v[i, i] = val_x * val_y

        for j in range(i + 1, l):
            G_j_x = gaussians[j].G_x(x, with_norm=True)
            G_j_y = gaussians[j].G_y(y, with_norm=True)

            val_x = scipy.integrate.simps(G_i_x * pot_x * G_j_x, x)
            val_y = scipy.integrate.simps(G_i_y * pot_y * G_j_y, y)

            v[i, j] = val_x * val_y
            v[j, i] = np.conjugate(val_x * val_y)

    return v


def construct_arbitrary_potential_elements(
    potential: callable, gaussians: list, x: np.array, y: np.array,
) -> np.ndarray:
    l = len(gaussians)
    v = np.zeros((l, l), dtype=type(potential(x[0], y[0])))
    err = np.zeros_like(v)

    for i in range(l):
        G_i = gaussians[i]

        for j in range(i, l):
            G_j = gaussians[j]

            func = (
                lambda y, x: G_i(x, y, with_norm=True)
                * potential(x, y)
                * G_j(x, y, with_norm=True)
            )

            val, err_val = scipy.integrate.dblquad(
                func, x[0], x[-1], y[0], y[-1]
            )

            v[i, j], err[i, j] = val, err_val

            if i != j:
                v[j, i], err[j, i] = np.conjugate(val), err_val

    return v
