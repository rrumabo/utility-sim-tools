import numpy as np

class BasePDESystem2D:
    def __init__(self, rhs_func):
        self.rhs_func = rhs_func

    def evolve(self, u0, dt, steps):
        raise NotImplementedError("Subclasses must implement evolve()")

class ExplicitPDESystem2D(BasePDESystem2D):
    def __init__(self, rhs_func):
        super().__init__(rhs_func)

    def evolve(self, u0, dt, steps):
        Nx, Ny = u0.shape
        u = u0.flatten()
        u_history = [u.copy()]

        for step in range(steps):
            rhs = self.rhs_func(u, step * dt)
            u = u + dt * rhs
            u_history.append(u.copy())

        u_history = np.array(u_history).reshape((steps+1, Nx, Ny))
        return u_history

def make_linear_rhs(operator, alpha=1.0):
    def rhs(u, t):
        return alpha * (operator @ u)
    return rhs