import numpy as np
import scipy.sparse as sp

def make_laplacian_2d(Nx, Ny, dx, dy):
    """
    Construct a sparse periodic 5-point finite-difference Laplacian
    on a 2D grid using Kronecker sums.

    Parameters:
        Nx (int): grid points in x
        Ny (int): grid points in y
        dx (float): grid spacing in x
        dy (float): grid spacing in y

    Returns:
        L (scipy.sparse.csr_matrix): sparse Laplacian of shape (Nx*Ny, Nx*Ny)
    """
    # 1D periodic Laplacian in x
    Dx = -2 * sp.eye(Nx, format="csr")
    Dx += sp.eye(Nx, k=1, format="csr") + sp.eye(Nx, k=-1, format="csr")
    Dx[0, -1] = Dx[-1, 0] = 1  # Periodic wrap
    Dx /= dx**2

    # 1D periodic Laplacian in y
    Dy = -2 * sp.eye(Ny, format="csr")
    Dy += sp.eye(Ny, k=1, format="csr") + sp.eye(Ny, k=-1, format="csr")
    Dy[0, -1] = Dy[-1, 0] = 1  # Periodic wrap
    Dy /= dy**2

    # Identity matrices
    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")

    # Kronecker sum: L = Iy ⊗ Dx + Dy ⊗ Ix
    L = sp.kron(Iy, Dx, format="csr") + sp.kron(Dy, Ix, format="csr")
    return L