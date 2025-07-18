import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def animate_heat_solution(x, u_history, dt=0.001, save_path="figures/heat_diffusion.gif"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, u_history[0], color='blue')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.1 * u_history.max())
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Heat Diffusion Over Time")

    def update(frame):
        line.set_ydata(u_history[frame])
        ax.set_title(f"Heat Diffusion at t = {frame * dt:.3f}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(u_history), interval=50)
    ani.save(save_path, writer="pillow", fps=20)
    plt.close()