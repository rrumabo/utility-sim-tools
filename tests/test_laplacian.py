import numpy as np
from numerics.laplacian_1d import make_laplacian_1d

def test_laplacian_annihilates_constant():
    N = 10
    dx = 1.0
    L = make_laplacian_1d(N, dx)
    constant_vec = np.ones(N)
    result = L @ constant_vec
    assert np.allclose(result, np.zeros(N)), "Laplacian should output 0 when applied to constant vector"