import numpy as np

def make_laplacian_2d(Nx, Ny, dx, dy):
    """
    Construct a periodic 5-point finite-difference Laplacian on a 2D grid
    using Kronecker sums for efficiency.

    Parameters:
        Nx (int): number of grid points in x
        Ny (int): number of grid points in y
        dx (float): spacing between points in x
        dy (float): spacing between points in y

    Returns:
        L (ndarray): Laplacian matrix of shape (Nx*Ny, Nx*Ny)
    """
    # 1D Laplacians
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    Dx = -2 * Ix + np.eye(Nx, k=1) + np.eye(Nx, k=-1)
    Dy = -2 * Iy + np.eye(Ny, k=1) + np.eye(Ny, k=-1)

    # Scale by grid spacing
    Dx /= dx**2
    Dy /= dy**2

    # 2D Laplacian via Kronecker sum
    L = np.kron(Iy, Dx) + np.kron(Dy, Ix)
    return L