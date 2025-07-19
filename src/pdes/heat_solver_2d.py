import numpy as np

def run_heat_solver_2d(u0, laplacian, alpha, dt, steps):
    u = u0.flatten()
    u_history = [u.copy()]
    
    for _ in range(steps):
        u = u + dt * alpha * laplacian @ u
        u_history.append(u.copy())
    
    return np.array(u_history)