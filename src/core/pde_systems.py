import numpy as np

from core.base_pde_system import BasePDESystem

class ExplicitPDESystem2D(BasePDESystem):
    def __init__(self, rhs_func):
        super().__init__(rhs_func)

class LinearPDESystem2D(BasePDESystem):
    def __init__(self, L_op, alpha=1.0, step_func=None):
        self.L_op = L_op
        self.alpha = alpha

        def rhs_func(u_flat, t):
            return self.alpha * (self.L_op @ u_flat)

        super().__init__(rhs_func, step_func)

def make_linear_rhs(operator, alpha=1.0):
    def rhs(u, t):
        return alpha * (operator @ u)
    return rhs

class LinearPDESystem1D(BasePDESystem):
    def __init__(self, L_op, alpha=1.0, step_func=None):
        self.L_op = L_op
        self.alpha = alpha

        def rhs_func(u_flat, t):
            return self.alpha * (self.L_op @ u_flat)

        super().__init__(rhs_func, step_func)