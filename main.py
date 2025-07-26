def run_simulation(cfg):
    return main(cfg)

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv

tracked_quantities = []

def diagnostics_fn(u, t):
    tracked_quantities.append({
        "t": t,
        "mass": np.sum(u),
        "l2": np.sqrt(np.sum(u**2))
    })

from src.utils.config_loader import load_config

from src.utils.diagnostics import plot_l2_error
from src.visualization.animation_1d import animate_heat_solution

def maybe_save_diagnostics(u_final, u_ref, dx, dy, folder):
    u_final = u_final.reshape(-1)
    u_ref = u_ref.reshape(-1)

    from src.utils.diagnostics import compute_l2_error
    l2_err = compute_l2_error(u_final, u_ref, dx, dy)

    os.makedirs(folder, exist_ok=True)
    # Save YAML summary
    with open(os.path.join(folder, "diagnostics.yaml"), "w") as f_yaml:
        yaml.dump({"L2_error": float(l2_err)}, f_yaml)

    # Save CSV version
    with open(os.path.join(folder, "diagnostics_summary.csv"), "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["L2_error"])
        writer.writeheader()
        writer.writerow({"L2_error": float(l2_err)})

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
    plt.savefig(f"{folder}/final_comparison.png", dpi=300)
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

    from src.core.rhs_examples import make_linear_rhs
    from src.core.time_integrators import rk4_step, euler_step
    if dim == 1:
        import numpy as _np
        from src.numerics.laplacian_1d import make_laplacian_1d
        from src.initial_conditions.profiles_1d import gaussian_bump as ic_func
        from src.core.pde_systems import LinearPDESystem1D

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        lap = make_laplacian_1d(N, dx)
        # Filter out 'type' key from init before passing as kwargs
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

    for step in range(steps):
        rhs_func = pde_system.rhs_func
        t = step * dt
        u = step_func(u, rhs_func, t, dt, diagnostics_fn)
        u_history.append(u.copy())
        diagnostics.append({
            "min": u.min(),
            "max": u.max(),
            "mean": u.mean(),
        })

    os.makedirs(out_cfg["folder"], exist_ok=True)

    if out_cfg.get("plot_profile", True):
        maybe_plot_final(x if dim == 1 else X[:,0],
                         u0 if dim == 1 else u0[:,u0.shape[1]//2],
                         u_history[-1],
                         out_cfg["folder"])

    if out_cfg.get("save_animation", True):
        animate_heat_solution(x if dim == 1 else (x, y),
                              u_history,
                              dt=dt,
                              save_path=f"{out_cfg['folder']}/heat_diffusion.gif")

    if out_cfg.get("save_diagnostics", True):
        print("DEBUG: u0 shape:", u0.shape)
        print("DEBUG: u_history[-1] shape:", u_history[-1].shape)
        maybe_save_diagnostics(u_history[-1], u0, dx, dx if dim == 1 else dx, out_cfg["folder"])
        with open(f"{out_cfg['folder']}/diagnostics_tracked.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t", "mass", "l2", "step", "min", "max", "mean"])
            writer.writeheader()
            for row in tracked_quantities:
                writer.writerow(row)
            for i, row in enumerate(diagnostics):
                writer.writerow({
                    "step": i,
                    "min": row["min"],
                    "max": row["max"],
                    "mean": row["mean"],
                })

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