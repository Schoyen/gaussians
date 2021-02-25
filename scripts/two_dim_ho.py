import time
import numpy as np

from gaussians import G2D
from gaussians.two_dim import construct_overlap_matrix_elements

import gaussians.two_dim_lib as tdl


omega = 1
l = 12

gaussians = [
    G2D((i, j), omega / 2) for i in range(l // 2) for j in range(l // 2)
]

t0 = time.time()
s = construct_overlap_matrix_elements(gaussians)
t1 = time.time()

print(f"Python time: {t1 - t0} sec")

t0 = time.time()
s_r = tdl.construct_overlap_matrix_elements([g.get_params() for g in gaussians])
t1 = time.time()

print(f"Rust time: {t1 - t0} sec")

np.testing.assert_allclose(s, s_r)
