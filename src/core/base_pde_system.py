class BasePDESystem:
    def __init__(self, rhs_func):
        """
        Base class for explicit PDE systems.
        
        Parameters:
            rhs_func: function u_flat, t → du/dt
        """
        self.rhs_func = rhs_func

    def evolve(self, u0, dt, steps):
        """
        Evolve the PDE system using forward Euler.

        Parameters:
            u0: initial state (2D array or flattened)
            dt: time step
            steps: number of steps

        Returns:
            history: list of solution snapshots
        """
        original_shape = u0.shape
        u = u0.ravel().copy()
        history = [u.copy()]

        for step in range(steps):
            rhs = self.rhs_func(u, step * dt)
            u = u + dt * rhs
            history.append(u.reshape(original_shape))

        return history