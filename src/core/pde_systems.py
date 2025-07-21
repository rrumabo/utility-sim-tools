import numpy as np

class LinearPDESystem2D:
    def __init__(self, operator, alpha=1.0, rhs_func=None):
        """
        Parameters:
            operator: callable or ndarray — linear spatial operator (matrix or function)
            alpha: float — scalar multiplier for main PDE term
            rhs_func: optional function f(u, t) — additional source term
        """
        self.operator = operator
        self.alpha = alpha
        self.rhs_func = rhs_func

    def evolve(self, u0, dt, steps, method="euler"):
        """
        Evolves the PDE in time using a chosen time integrator.

        Parameters:
            u0 (2D ndarray): Initial condition
            dt (float): Time step
            steps (int): Number of steps
            method (str): Time integrator ("euler" for now)

        Returns:
            u_history: ndarray of shape (steps+1, Nx, Ny)
        """
        Nx, Ny = u0.shape
        u = u0.flatten()
        u_history = [u.copy()]

        for step in range(steps):
            rhs = self.alpha * (self.operator @ u)
            if self.rhs_func:
                rhs += self.rhs_func(u, step * dt)
            u = u + dt * rhs
            u_history.append(u.copy())

        u_history = np.array(u_history).reshape((steps+1, Nx, Ny))
        return u_history