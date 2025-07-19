import numpy as np

def run_heat_solver_1d(u0, laplacian, alpha, dt, steps):
    u = u0.copy()
    u_history = [u.copy()]
    diagnostics = []

    for _ in range(steps):
        u = u + dt * alpha * laplacian @ u
        u_history.append(u.copy())

        diagnostics.append({
            "min": np.min(u),
            "max": np.max(u),
            "mean": np.mean(u),
        })

    return np.array(u_history), diagnostics