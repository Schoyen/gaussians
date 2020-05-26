import numpy as np
import matplotlib.pyplot as plt
import scipy.special

from gaussians.two_dim.coulomb_elements import extended_bessel


sigma = lambda p, q: (p + q) / (4 * p * q)
delta = np.linspace(-1000, 1000, 8001)
delta_vec = np.array([delta, np.zeros_like(delta)])


def extended_bessel_2(n, sigma, delta):
    # Old implementation, this is unstable
    delta_sq = delta[0] ** 2 + delta[1] ** 2
    arg = -delta_sq / (8 * sigma)

    return np.exp(arg) * scipy.special.iv(n, arg)


for n in range(10):
    plt.plot(
        delta, extended_bessel(n, sigma(1, 1), delta_vec), label=f"n = {n}"
    )
    plt.plot(
        delta,
        extended_bessel_2(n, sigma(1, 1), delta_vec),
        label=f"(2) n = {n}",
    )

plt.legend()
plt.grid()
plt.show()
