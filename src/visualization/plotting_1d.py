import matplotlib.pyplot as plt
import os

def plot_initial_final(x, u0, u_final, title="Initial vs Final", filename=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, label="Initial", linestyle="--")
    plt.plot(x, u_final, label="Final", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()