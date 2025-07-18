import numpy as np
from scipy.integrate import simpson

def compute_l2_error(u_numeric, u_reference, dx):
    """Compute L² error between two functions."""
    diff_sq = (u_numeric - u_reference)**2
    return np.sqrt(simpson(diff_sq, dx=dx))