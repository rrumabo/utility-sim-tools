import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.numerics.laplacian_1d import make_laplacian_1d
from src.pdes.heat_solver_1d import run_heat_solver_1d
from src.utils.diagnostics import compute_l2_error
from src.visualization.animation_1d import animate_heat_solution

def gaussian_bump(x, center=0.0, width=0.5, amplitude=1.0):
    return amplitude * np.exp(-((x - center) / width)**2)

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulation"]
    init = cfg["initial_condition"]
    out_cfg = cfg["output"]

    L = sim["L"]
    N = sim["N"]
    dx = L / N
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

    laplacian = make_laplacian_1d(N, dx)

    if init["type"] == "gaussian_bump":
        u0 = gaussian_bump(x, center=init["center"], width=init["width"], amplitude=init["amplitude"])
    else:
        raise ValueError("Unsupported initial condition")

    u_history = run_heat_solver_1d(u0, laplacian, sim["alpha"], sim["dt"], sim["steps"])

    os.makedirs(out_cfg["folder"], exist_ok=True)

    if out_cfg.get("plot_profile", True):
        plt.figure(figsize=(8, 4))
        plt.plot(x, u0, label="Initial u₀", linestyle="--")
        plt.plot(x, u_history[-1], label="Final u", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("Initial vs Final Heat Profile")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out_cfg['folder']}/final_comparison.png", dpi=300)
        plt.close()

    if out_cfg.get("save_animation", True):
        animate_heat_solution(x, u_history, dt=sim["dt"], save_path=f"{out_cfg['folder']}/heat_diffusion.gif")

    if out_cfg.get("save_diagnostics", True):
        l2_err = compute_l2_error(u_history[-1], u0, dx)
        with open(f"{out_cfg['folder']}/diagnostics.yaml", "w") as f:
            yaml.dump({"L2_error": float(l2_err)}, f)

if __name__ == "__main__":
    main()