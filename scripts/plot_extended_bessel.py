import numpy as np
import matplotlib.pyplot as plt

from gaussians.two_dim.coulomb_elements import extended_bessel


sigma = lambda p, q: (p + q) / (4 * p * q)
delta = np.linspace(-50, 50, 8001)
delta_vec = np.array([delta, np.zeros_like(delta)])


for n in range(10):
    plt.plot(
        delta, extended_bessel(n, sigma(1, 1), delta_vec), label=f"n = {n}"
    )

plt.legend()
plt.grid()
plt.show()
