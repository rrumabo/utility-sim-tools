def compute_l2_error(u_final, u_ref, dx, dy=1.0):
    """Compute the L2 error between two arrays."""
    return np.sqrt(np.sum((u_final - u_ref) ** 2) * dx * dy)

import matplotlib.pyplot as plt
import numpy as np

def plot_mass_evolution(u_history, dx, dy, title="Mass over time"):
    mass = [np.sum(u) * dx * dy for u in u_history]
    plt.plot(mass)
    plt.xlabel("Time step")
    plt.ylabel("Total mass")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_l2_error(u_history, reference, dx, dy, title="L² Error over Time"):
    errors = [
        np.sqrt(np.sum((u.reshape(reference.shape) - reference)**2) * dx * dy)
        for u in u_history
    ]

    plt.plot(errors)
    plt.xlabel("Time step")
    plt.ylabel("L² error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()