import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_solutions(
    x, dict: dict, x_label: str, y_label: str, title: str, filename: str
):
    x = jnp.asarray(x)
    plt.figure(figsize=(8, 4))
    for label, y in dict.items():
        plt.plot(x, jnp.asarray(y), label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_field(
    X,
    Y,
    U,
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    cmap: str = "RdBu_r",
    cbar_label: str = "u",
):
    """Heatmap of a 2-D field U on the grid (X, Y). All three are (Nx, Ny)."""
    plt.figure(figsize=(7, 4))
    mesh = plt.pcolormesh(
        jnp.asarray(X), jnp.asarray(Y), jnp.asarray(U), shading="auto", cmap=cmap
    )
    plt.colorbar(mesh, label=cbar_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_bands(
    histories: dict,
    filename: str,
    title: str = "Training loss",
    band: str = "iqr",
):
    """
    Loss curves over several seeds, drawn as a central line with a spread band
    """
    plt.figure(figsize=(8, 4))
    for label, hist in histories.items():
        h = jnp.atleast_2d(jnp.asarray(hist))
        steps = jnp.arange(h.shape[1])
        if band == "std":
            log_h = jnp.log10(jnp.maximum(h, 1e-30))
            mean, std = log_h.mean(axis=0), log_h.std(axis=0)
            mid, low, high = 10**mean, 10 ** (mean - std), 10 ** (mean + std)
        else:
            low, mid, high = jnp.percentile(h, jnp.array([25.0, 50.0, 75.0]), axis=0)
        (line,) = plt.plot(steps, mid, label=label)
        if h.shape[0] > 1:
            plt.fill_between(steps, low, high, alpha=0.25, color=line.get_color())
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_losses(histories: dict, filename: str, title: str = "Training loss"):
    plt.figure(figsize=(8, 4))
    for label, hist in histories.items():
        plt.plot(jnp.asarray(hist), label=label)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
