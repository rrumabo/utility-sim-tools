import numpy as np
import scipy.sparse as sp

def make_laplacian_2d(Nx, Ny, dx, dy):
    """
    Construct the 2D Laplacian using Kronecker sums with periodic boundary conditions.

    Returns:
        L (csr_matrix): Sparse Laplacian matrix of shape (Nx*Ny, Nx*Ny)
    """
    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")

    # Use LIL for safe element assignment
    Dx = sp.lil_matrix((Nx, Nx))
    for i in range(Nx):
        Dx[i, i] = -2.0
        Dx[i, (i + 1) % Nx] = 1.0
        Dx[i, (i - 1) % Nx] = 1.0
    Dx /= dx**2

    Dy = sp.lil_matrix((Ny, Ny))
    for j in range(Ny):
        Dy[j, j] = -2.0
        Dy[j, (j + 1) % Ny] = 1.0
        Dy[j, (j - 1) % Ny] = 1.0
    Dy /= dy**2

    # Convert to CSR for efficient use
    Dx = Dx.tocsr()
    Dy = Dy.tocsr()

    # Kronecker sum
    L = sp.kron(Iy, Dx, format="csr") + sp.kron(Dy, Ix, format="csr")
    return L