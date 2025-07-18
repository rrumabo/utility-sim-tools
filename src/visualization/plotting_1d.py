import matplotlib.pyplot as plt
import os

def plot_initial_final(x, u0, u_final, save_path="figures/final_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, label="Initial $u_0$", linestyle="--")
    plt.plot(x, u_final, label="Final $u(x, t)$", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Initial vs Final Heat Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()