import numpy as np

def run_heat_solver_2d(u0, laplacian, alpha, dt, steps):
    Nx, Ny = u0.shape
    u = u0.flatten()
    u_history = [u.copy()]
    diagnostics = []
    
    for _ in range(steps):
        u = u + dt * alpha * laplacian @ u
        u_history.append(u.copy())
        diagnostics.append({
            "min": float(u.min()),
            "max": float(u.max()),
            "mean": float(u.mean()),
        })
    
    u_history = np.array(u_history)            # shape (steps+1, N)
    u_history = u_history.reshape((steps+1, Nx, Ny))
    return u_history, diagnostics