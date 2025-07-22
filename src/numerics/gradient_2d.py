import numpy as np

def make_gradient_2d(Nx, Ny, dx, dy):
    """
    Returns a function that computes the gradient of a flattened u field
    using central differences and periodic boundaries.
    """
    def gradient(u_flat):
        u = u_flat.reshape(Nx, Ny)

        # Central difference (periodic BCs)
        dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)

        return dudx.flatten()[:, None], dudy.flatten()[:, None]

    return gradient