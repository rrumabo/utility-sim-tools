import numpy as np
from src.numerics.laplacian_2d import make_laplacian_2d

def test_laplacian_2d_annihilates_constant():
    Nx, Ny = 10, 10
    dx = dy = 1.0
    L = make_laplacian_2d(Nx, Ny, dx, dy)
    constant_field = np.ones((Nx, Ny)).flatten()
    result = L @ constant_field
    assert np.allclose(result, np.zeros(Nx * Ny)), "2D Laplacian should output 0 for constant field"