import time
import numpy as np

from gaussians import G2D
from gaussians.two_dim import (
    construct_overlap_matrix_elements,
    construct_kinetic_matrix_elements,
    construct_multipole_moment_matrix_elements,
    construct_coulomb_matrix_elements,
)

import gaussians.two_dim_lib as tdl


omega = 1

gaussians = [
    G2D((0, 0), omega / 2),
    G2D((0, 1), omega / 2),
    G2D((1, 0), omega / 2),
    G2D((2, 0), omega / 2),
    G2D((1, 1), omega / 2),
    G2D((0, 2), omega / 2),
]
g_params = [g.get_params() for g in gaussians]


def time_func(func, *args, lang="Python"):
    t0 = time.time()
    mat = func(*args)
    t1 = time.time()

    print(f"{lang} {func.__name__}: {t1 - t0} sec")

    return mat


s = time_func(construct_overlap_matrix_elements, gaussians)
s_r = time_func(tdl.construct_overlap_matrix_elements, g_params, lang="Rust")

np.testing.assert_allclose(s, s_r)

t = time_func(construct_kinetic_matrix_elements, gaussians)
t_r = time_func(
    tdl.construct_kinetic_operator_matrix_elements, g_params, lang="Rust"
)

np.testing.assert_allclose(t, t_r)

v_x = time_func(
    construct_multipole_moment_matrix_elements,
    *[(2, 0), (0, 0), gaussians],
    lang="Python (x)",
)
v_y = time_func(
    construct_multipole_moment_matrix_elements,
    *[(0, 2), (0, 0), gaussians],
    lang="Python (y)",
)
v_rx = time_func(
    tdl.construct_multipole_moment_matrix_elements,
    *[(2, 0), (0, 0), g_params],
    lang="Rust (x)",
)
v_ry = time_func(
    tdl.construct_multipole_moment_matrix_elements,
    *[(0, 2), (0, 0), g_params],
    lang="Rust (y)",
)

v = 0.5 * omega ** 2 * (v_x + v_y)
v_r = 0.5 * omega ** 2 * (v_rx + v_ry)

np.testing.assert_allclose(v, v_r)

print("Starting Coulomb computations")
print(f"Num Gaussians: {len(gaussians)}")
u = time_func(construct_coulomb_matrix_elements, gaussians)
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

np.testing.assert_allclose(u, u_r)

gaussians.append(G2D((0, 0), omega / 2, (2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((0, 1), omega / 2, (2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((1, 0), omega / 2, (2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((2, 0), omega / 2, (2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((1, 1), omega / 2, (2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((0, 2), omega / 2, (2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((0, 0), omega / 2, (-2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((0, 1), omega / 2, (-2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((1, 0), omega / 2, (-2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((2, 0), omega / 2, (-2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((1, 1), omega / 2, (-2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)

gaussians.append(G2D((0, 2), omega / 2, (-2, 0)))
g_params = [g.get_params() for g in gaussians]

print(f"Num Gaussians: {len(gaussians)}")
u_r = time_func(
    tdl.construct_coulomb_operator_matrix_elements, g_params, lang="Rust"
)
