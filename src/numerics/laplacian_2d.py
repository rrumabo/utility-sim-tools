import numpy as np

def make_laplacian_2d(Nx, Ny, dx, dy):
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    
    Lx = -2 * Ix + np.eye(Nx, k=1) + np.eye(Nx, k=-1)
    Ly = -2 * Iy + np.eye(Ny, k=1) + np.eye(Ny, k=-1)
    
    Lx /= dx**2
    Ly /= dy**2

    return np.kron(Iy, Lx) + np.kron(Ly, Ix)