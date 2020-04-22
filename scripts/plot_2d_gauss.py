import numpy as np
import matplotlib.pyplot as plt

from gaussians import G2D


x = np.linspace(-3, 3, 201)

X, Y = np.meshgrid(x, x)

g2d = G2D((0, 1), 1, [0.3, -0.2])

plt.contourf(X, Y, g2d(X, Y), levels=20)
plt.colorbar()
plt.show()
