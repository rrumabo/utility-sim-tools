import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_2d(u_history, x, y, interval=30, filename=None, cmap="inferno"):
    """
    Create a 2D animation of u(x, y, t) over time.

    Parameters:
        u_history (ndarray): shape (T, Nx, Ny)
        x, y (1D arrays): spatial grid vectors
        interval (int): frame interval in ms
        filename (str or None): if given, saves as .mp4
        cmap (str): matplotlib colormap

    Returns:
        HTML5 video animation (for notebook use)
    """
    from IPython.display import HTML

    T, Nx, Ny = u_history.shape
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots()
    vmin = np.min(u_history[0])
    vmax = np.max(u_history[0])
    im = ax.imshow(u_history[0], extent=[x.min(), x.max(), y.min(), y.max()],
                   origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Time step: 0")
    fig.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(u_history[frame])
        ax.set_title(f"Time step: {frame}")
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)

    if filename:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=1000 // interval)
        anim.save(filename, writer=writer)
        plt.close(fig)
        return None
    else:
        plt.close(fig)
        return HTML(anim.to_html5_video())