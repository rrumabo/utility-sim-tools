import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import csv

from src.numerics.laplacian_1d import make_laplacian_1d
from src.pdes.heat_solver_1d import run_heat_solver_1d
from src.utils.diagnostics import compute_l2_error
from src.visualization.animation_1d import animate_heat_solution

from src.initial_conditions.profiles_1d import gaussian_bump, square_pulse, triangle_wave

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim = cfg["simulation"]
    init = cfg["initial_condition"]
    out_cfg = cfg["output"]

    L = sim["L"]
    N = sim["N"]
    dx = L / N
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

    laplacian = make_laplacian_1d(N, dx)

    ic_type = init["type"]
    kwargs = {
        "center": init["center"],
        "width": init["width"],
        "amplitude": init["amplitude"]
    }

    if ic_type == "gaussian_bump":
        u0 = gaussian_bump(x, **kwargs)
    elif ic_type == "square_pulse":
        u0 = square_pulse(x, **kwargs)
    elif ic_type == "triangle_wave":
        u0 = triangle_wave(x, **kwargs)
    else:
        raise ValueError(f"Unsupported initial condition: {ic_type}")

    u_history, diagnostics = run_heat_solver_1d(
    u0, laplacian, sim["alpha"], sim["dt"], sim["steps"]
    )

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

        import csv
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
    args = parser.parse_args()
    main(args.config)