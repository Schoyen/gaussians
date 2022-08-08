import numpy as np


def theta_0(k_x, k_y):
    return np.arctan(-k_x / k_y)


def fourier_coulomb_1(k_x, k_y):
    theta = theta_0(k_x, k_y)

    return 1 / np.abs(-k_x * np.sin(theta) + k_y * np.cos(theta))


def fourier_coulomb_2(k_x, k_y):
    return 1 / np.sqrt(k_x**2 + k_y**2)


def fourier_coulomb_3(k_x, k_y):
    theta = theta_0(k_x, k_y)

    return 1 / np.sqrt(
        k_x**2 * np.sin(theta) ** 2
        + k_y**2 * np.cos(theta) ** 2
        - 2 * k_x * k_y * np.sin(theta) * np.cos(theta)
    )


def fourier_coulomb_4(k_x, k_y):
    theta = theta_0(k_x, k_y)

    return 1 / np.sqrt(
        k_x**2 * np.sin(theta) ** 2
        + k_y**2 * np.cos(theta) ** 2
        + 2 * k_y**2 * np.sin(theta) ** 2
    )


for k_x in np.linspace(0.1, 2 * np.pi - 0.1, 10):
    for k_y in np.linspace(0.1, 2 * np.pi - 0.1, 10):
        fourier_1 = fourier_coulomb_1(k_x, k_y)
        fourier_2 = fourier_coulomb_2(k_x, k_y)
        fourier_3 = fourier_coulomb_3(k_x, k_y)
        fourier_4 = fourier_coulomb_4(k_x, k_y)

        for f_1 in [fourier_1, fourier_2, fourier_3, fourier_4]:
            for f_2 in [fourier_1, fourier_2, fourier_3, fourier_4]:
                assert abs(f_1 - f_2) < 1e-12
