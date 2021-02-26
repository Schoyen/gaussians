from gaussians.two_dim.coulomb_elements import extended_bessel

delta = [2.3, 0.7]
sigma = 0.6

for n in range(5):
    print(f"I_{n} exp: {extended_bessel(n, sigma, delta)}")
