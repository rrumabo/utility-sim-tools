import numpy as np

def run_heat_solver_2d(u0, laplacian, alpha, dt, steps):
    Nx, Ny = u0.shape
    assert laplacian.shape == (Nx * Ny, Nx * Ny), "Laplacian size mismatch"
    u = u0.flatten()
    u_history = [u.copy()]
    diagnostics = []

    def euler_step(u, dt, laplacian, alpha):
        return u + dt * alpha * (laplacian @ u)

    for _ in range(steps):
        u = euler_step(u, dt, laplacian, alpha)
        u_history.append(u.copy())
        diagnostics.append({
            "min": float(u.min()),
            "max": float(u.max()),
            "mean": float(u.mean()),
        })

    u_history = np.array(u_history)            # shape (steps+1, N)
    u_history = u_history.reshape((steps+1, Nx, Ny))
    return u_history, diagnostics