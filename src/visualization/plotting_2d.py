import matplotlib.pyplot as plt
import os
from matplotlib import animation

def plot_final_frame(u, x, y, title="Final State", cmap="viridis", save_path=None):
    plt.imshow(u, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_initial_vs_final(u0, u1, x, y, titles=("Initial", "Final"), cmap="viridis"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(u0, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap)
    axes[0].set_title(titles[0])
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u1, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap)
    axes[1].set_title(titles[1])
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def animate_2d(u_history, x, y, interval=40, filename="figures/heat_2D.mp4", cmap="viridis"):
    if not os.path.exists("figures"):
        os.makedirs("figures")

    fig, ax = plt.subplots()
    img = ax.imshow(u_history[0], extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap=cmap)

    def update(frame):
        img.set_data(u_history[frame])
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(u_history), interval=interval, blit=True)

    if filename:
        anim.save(filename, writer="ffmpeg")
        plt.close(fig)
    else:
        plt.show()