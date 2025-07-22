import numpy as np
import scipy.sparse as sp

def make_laplacian_1d(N, dx):
    lap = sp.lil_matrix((N, N))
    for i in range(N):
        lap[i, i] = -2.0
        lap[i, (i - 1) % N] = 1.0
        lap[i, (i + 1) % N] = 1.0
    return lap.tocsr() / dx**2
