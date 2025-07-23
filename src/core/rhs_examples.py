import numpy as np

def make_nlse_rhs(L_op, alpha=1.0, beta=1.0):
    def rhs(u_flat, t):
        u = u_flat.reshape(L_op.shape[0], 1)
        linear_term = alpha * (L_op @ u)
        nonlinear_term = beta * np.abs(u)**2 * u
        return (linear_term + nonlinear_term).flatten()
    return rhs

def make_linear_rhs(operator, alpha=1.0):
    """
    Generic linear RHS for PDEs of the form du/dt = α * (operator @ u)
    """
    def rhs(u, t):
        return alpha * (operator @ u)
    return rhs

def make_burgers_rhs(L_op, grad_func, nu=0.1):
    """
    Burgers'-style PDE: du/dt = -u · ∇u + ν ∇²u

    Parameters:
        L_op: Laplacian matrix (Nx*Ny x Nx*Ny)
        grad_func: function u_flat → (∂u/∂x, ∂u/∂y)
        nu: viscosity

    Returns:
        rhs(u, t)
    """
    def rhs(u_flat, t):
        u = u_flat.reshape(L_op.shape[0], 1)
        grad_x, grad_y = grad_func(u)
        nonlinear = u * grad_x + u * grad_y  # crude estimate of u · ∇u
        linear = nu * (L_op @ u)
        return (-nonlinear + linear).flatten()
    return rhs