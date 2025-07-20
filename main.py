import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

import argparse
import csv

from src.utils.diagnostics import compute_l2_error
from src.visualization.animation_1d import animate_heat_solution

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def maybe_save_diagnostics(u_final, u_ref, dx, folder):
    l2_err = compute_l2_error(u_final, u_ref, dx)
    with open(f"{folder}/diagnostics.yaml", "w") as f:
        yaml.dump({"L2_error": float(l2_err)}, f)

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
    sim = cfg["simulation"]
    init = cfg["initial_condition"]
    out_cfg = cfg["output"]

    L = sim["L"]
    N = sim["N"]
    dx = L / N

    if dim == 1:
        import numpy as _np
        from src.numerics.laplacian_1d import make_laplacian_1d
        from src.pdes.heat_solver_1d import run_heat_solver_1d
        from src.initial_conditions.profiles_1d import gaussian_bump as ic_func

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        lap = make_laplacian_1d(N, dx)
        u0 = ic_func(x, center=init["center"], width=init["width"], amplitude=init["amplitude"])
        u_history, diagnostics = run_heat_solver_1d(u0, lap, sim["alpha"], sim["dt"], sim["steps"])

    elif dim == 2:
        from src.numerics.laplacian_2d import make_laplacian_2d
        from src.pdes.heat_solver_2d import run_heat_solver_2d
        from src.initial_conditions.gaussian_2d import gaussian_bump_2d as ic_func

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        y = x.copy()
        X, Y = np.meshgrid(x, y, indexing="ij")
        lap = make_laplacian_2d(N, N, dx, dx)
        u0 = ic_func(X, Y, center=(init["center"], init["center"]), width=init["width"], amplitude=init["amplitude"])
        u_history, diagnostics = run_heat_solver_2d(u0, lap, sim["alpha"], sim["dt"], sim["steps"])

    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    os.makedirs(out_cfg["folder"], exist_ok=True)

    if out_cfg.get("plot_profile", True):
        maybe_plot_final(x if dim == 1 else X[:,0], 
                         u0 if dim == 1 else u0[:,u0.shape[1]//2], 
                         u_history[-1], 
                         out_cfg["folder"])

    if out_cfg.get("save_animation", True):
        animate_heat_solution(x if dim == 1 else (x, y), 
                              u_history, 
                              dt=sim["dt"], 
                              save_path=f"{out_cfg['folder']}/heat_diffusion.gif")

    if out_cfg.get("save_diagnostics", True):
        maybe_save_diagnostics(u_history[-1], u0, dx, out_cfg["folder"])
        with open(f"{out_cfg['folder']}/diagnostics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "min", "max", "mean"])
            writer.writeheader()
            for i, row in enumerate(diagnostics):
                writer.writerow({
                    "step": i,
                    "min": row["min"],
                    "max": row["max"],
                    "mean": row["mean"],
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heat solver with configuration")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--no-animation", action="store_true", help="Disable saving animation")
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable saving diagnostics")
    parser.add_argument("--no-profile", action="store_true", help="Disable final profile plot")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_animation:
        cfg["output"]["save_animation"] = False
    if args.no_diagnostics:
        cfg["output"]["save_diagnostics"] = False
    if args.no_profile:
        cfg["output"]["plot_profile"] = False

    main(cfg)