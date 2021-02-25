import time
import numpy as np

from gaussians import G1D
from gaussians.one_dim import construct_overlap_matrix_elements


import gaussians.one_dim_lib as odl


omega = 1
l = 6

gaussians = [G1D(i, omega / 2) for i in range(l)]

t0 = time.time()
s = construct_overlap_matrix_elements(gaussians)
t1 = time.time()

print(f"Python time: {t1 - t0} sec")

t0 = time.time()
s_r = odl.construct_overlap_matrix_elements([g.get_params() for g in gaussians])
t1 = time.time()

print(f"Rust time: {t1 - t0} sec")

np.testing.assert_allclose(s, s_r)
