import os
import yaml
import csv
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


# --- New diagnostics saving functions ---
def save_diagnostics_summary(u_final, u_ref, dx, dy, path):
    """Save final L2 error and field extrema to a YAML file."""
    if not os.path.exists(path):
        os.makedirs(path)

    l2_err = compute_l2_error(u_final, u_ref, dx, dy)
    diagnostics = {
        "L2_error": float(l2_err),
        "min": float(np.min(u_final)),
        "max": float(np.max(u_final)),
        "mean": float(np.mean(u_final)),
    }

    with open(os.path.join(path, "diagnostics.yaml"), "w") as f:
        yaml.dump(diagnostics, f)


def save_diagnostics_csv(diagnostics_list, path, filename="diagnostics.csv"):
    """Save a list of diagnostic dictionaries to a CSV file."""
    if not diagnostics_list:
        return

    if not os.path.exists(path):
        os.makedirs(path)

    keys = diagnostics_list[0].keys()
    with open(os.path.join(path, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in diagnostics_list:
            writer.writerow(row)