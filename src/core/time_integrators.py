def euler_step(u, rhs_func, t, dt, diagnostics_fn=None):
    if diagnostics_fn:
        diagnostics_fn(u, t)
    return u + dt * rhs_func(u, t)


def rk4_step(u, rhs_func, t, dt, diagnostics_fn=None):
    if diagnostics_fn:
        diagnostics_fn(u, t)
    k1 = rhs_func(u, t)
    k2 = rhs_func(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_func(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_func(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)