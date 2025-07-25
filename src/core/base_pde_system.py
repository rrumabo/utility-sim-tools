from utils.diagnostic_manager import DiagnosticManager
class BasePDESystem:
    def __init__(self, rhs_func, step_func, diagnostic_manager=None):
        self.rhs_func = rhs_func
        self.step_func = step_func
        self.diagnostic_manager = diagnostic_manager

    def evolve(self, u0, dt, steps):
        u = u0.copy()
        u_history = [u.copy()]

        for step in range(steps):
            t = step * dt
            u = self.step_func(u, self.rhs_func, t, dt)
            u_history.append(u.copy())

        return u_history