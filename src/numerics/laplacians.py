import numpy as np

def make_laplacian_1d(N, dx):
    lap = np.zeros((N, N))
    for i in range(N):
        lap[i, i] = -2.0
        lap[i, (i - 1) % N] = 1.0
        lap[i, (i + 1) % N] = 1.0
    return lap / dx**2
