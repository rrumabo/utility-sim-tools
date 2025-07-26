def run_simulation(cfg):
    return main(cfg)

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv

from src.utils.diagnostic_manager import DiagnosticManager

from src.utils.config_loader import load_config

 
from src.visualization.animation_1d import animate_heat_solution

def maybe_plot_final(x, u0, u_final, folder):
    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, label="Initial u₀", linestyle="--")
    plt.plot(x, u_final, label="Final u", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Initial vs Final Heat Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "final_comparison.png"), dpi=300)
    plt.close()

def main(cfg):
    dim = cfg.get("dimension", 1)
    init = cfg["initial_condition"]
    out_cfg = cfg["output"]

    pde_cfg = cfg["pde"]
    integrator_cfg = cfg["integrator"]

    L = cfg["grid"]["L"]
    N = cfg["grid"]["N"]
    dx = L / N
    dy = dx  # Assume square grid by default

    from src.core.rhs_examples import make_linear_rhs
    from src.core.time_integrators import rk4_step, euler_step
    if dim == 1:
        import numpy as _np
        from src.numerics.laplacian_1d import make_laplacian_1d
        from src.initial_conditions.profiles_1d import gaussian_bump as ic_func
        from src.core.pde_systems import LinearPDESystem1D

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        lap = make_laplacian_1d(N, dx)
        
        ic_params = {k: v for k, v in init.items() if k != "type"}
        u0 = ic_func(x, **ic_params)
        u0 = u0.reshape(-1)

        pde_system = LinearPDESystem1D(lap, alpha=pde_cfg["alpha"], step_func=None)

    elif dim == 2:
        from src.numerics.laplacian_2d import make_laplacian_2d
        from src.initial_conditions.gaussian_2d import gaussian_bump_2d as ic_func
        from src.core.pde_systems import LinearPDESystem2D

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        y = x.copy()
        X, Y = np.meshgrid(x, y, indexing="ij")
        lap = make_laplacian_2d(N, N, dx, dx)
        u0 = ic_func(X, Y, center=(init["center"], init["center"]), width=init["width"], amplitude=init["amplitude"])
        u0 = u0.reshape(-1)  

        pde_system = LinearPDESystem2D(lap, alpha=pde_cfg["alpha"], step_func=None)

    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    if integrator_cfg["method"] == "rk4":
        step_func = rk4_step
    elif integrator_cfg["method"] == "euler":
        step_func = euler_step
    else:
        raise ValueError(f"Unsupported integrator method: {integrator_cfg['method']}")

    dt = cfg["time"]["dt"]
    steps = cfg["time"]["steps"]

    u = u0.copy()
    u_history = [u.copy()]
    diagnostics = []
    diagnostics_manager = DiagnosticManager(dx=dx, dy=dy if dim == 2 else None, u_ref=u0)

    for step in range(steps):
        rhs_func = pde_system.rhs_func
        t = step * dt
        u = step_func(u, rhs_func, t, dt)
        u_history.append(u.copy())
        diagnostics.append({
            "step": step,
            "min": u.min(),
            "max": u.max(),
            "mean": u.mean(),
        })
        diagnostics_manager.track_step(u, step)

    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), out_cfg["folder"]))
    os.makedirs(output_folder, exist_ok=True)

    if out_cfg.get("plot_profile", True):
        maybe_plot_final(x if dim == 1 else X[:,0],
                         u0 if dim == 1 else u0[:,u0.shape[1]//2],
                         u_history[-1],
                         output_folder)

    if out_cfg.get("save_animation", True):
        animate_heat_solution(x if dim == 1 else (x, y),
                              u_history,
                              dt=dt,
                              save_path=os.path.join(output_folder, "heat_diffusion.gif"))

    if out_cfg.get("save_diagnostics", True):
        diagnostics_manager.save_csv(os.path.join(output_folder, "diagnostics_tracked.csv"))
        diagnostics_manager.save_yaml(os.path.join(output_folder, "diagnostics_summary.yaml"))

    return u_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heat solver with configuration")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--no-animation", action="store_true", help="Disable saving animation")
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable saving diagnostics")
    parser.add_argument("--no-profile", action="store_true", help="Disable final profile plot")
    parser.add_argument("--pde", type=str, help="Override PDE type (e.g. heat, nlse, burgers)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pde:
        cfg["pde"]["type"] = args.pde
    if args.no_animation:
        cfg["output"]["save_animation"] = False
    if args.no_diagnostics:
        cfg["output"]["save_diagnostics"] = False
    if args.no_profile:
        cfg["output"]["plot_profile"] = False

    main(cfg)