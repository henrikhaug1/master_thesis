import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_solution_grid(
    x,
    exact,
    predictions: dict,
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    ncols: int = 2,
):
    """One panel per model: the exact solution in red, that model's in dashed blue.

    Overlaying every model on one axis hides the small deviations that matter
    once they all roughly fit; a panel each keeps the comparison against the
    exact curve readable. Axes are shared, so the panels stay comparable.
    """
    x = jnp.asarray(x)
    exact = jnp.asarray(exact)
    n = len(predictions)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = axes.ravel()

    for i, (label, y) in enumerate(predictions.items()):
        ax = flat[i]
        ax.plot(x, exact, color="red", linewidth=1.8, label="exact")
        ax.plot(
            x,
            jnp.asarray(y),
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label="prediction",
        )
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if i >= n - ncols:  # bottom-most panel of its column
            ax.set_xlabel(x_label)
        if i % ncols == 0:
            ax.set_ylabel(y_label)

    for ax in flat[n:]:  # unused cells in a ragged grid
        ax.axis("off")

    flat[0].legend(loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def plot_fields(
    X,
    Y,
    U_exact,
    fields: dict,
    x_label: str,
    y_label: str,
    title: str,
    fig_dir,
    cbar_label: str = "u",
):
    """One PDF each for the exact field, every model's field and its absolute error.

    Files are named <title>_field_<name>.pdf and <title>_error_<name>.pdf, with
    spaces in the title as underscores.
    """
    slug = title.replace(" ", "_")
    for name, U in [("exact", U_exact), *fields.items()]:
        plot_field(
            X,
            Y,
            U,
            x_label,
            y_label,
            f"{title} - {name}",
            f"{fig_dir}/{slug}_field_{name}.pdf",
            cbar_label=cbar_label,
        )
    for name, U in fields.items():
        plot_field(
            X,
            Y,
            jnp.abs(U - U_exact),
            x_label,
            y_label,
            f"{title} - |{name} - exact|",
            f"{fig_dir}/{slug}_error_{name}.pdf",
            cmap="magma",
            cbar_label="abs. error",
        )


def plot_loss_bands(
    histories: dict,
    filename: str,
    title: str = "Training loss",
    band: str = "iqr",
    steps=None,
    y_label: str = "loss",
    references: dict | None = None,
):
    """
    Loss curves over several seeds, drawn as a central line with a spread band.

    Histories that stopped early are NaN-padded to a common length, so every
    reduction here is nan-aware: past the first stop the band is taken over the
    seeds still running, and it ends where the last one stopped.

    `references` maps a label to a constant value drawn as a dashed horizontal
    line, for methods without a training history such as an FEM solution.
    """
    plt.figure(figsize=(8, 4))
    for label, hist in histories.items():
        h = jnp.atleast_2d(jnp.asarray(hist))
        if steps is None:
            x = jnp.arange(h.shape[1])
        else:
            x = jnp.asarray(steps[label] if isinstance(steps, dict) else steps)
        if band == "std":
            log_h = jnp.log10(jnp.maximum(h, 1e-30))
            mean, std = jnp.nanmean(log_h, axis=0), jnp.nanstd(log_h, axis=0)
            mid, low, high = 10**mean, 10 ** (mean - std), 10 ** (mean + std)
        else:
            low, mid, high = jnp.nanpercentile(h, jnp.array([25.0, 50.0, 75.0]), axis=0)
        (line,) = plt.plot(x, mid, label=label)
        if h.shape[0] > 1:
            plt.fill_between(x, low, high, alpha=0.25, color=line.get_color())
    for label, value in (references or {}).items():
        plt.axhline(value, linestyle="--", linewidth=1.2, color="black", label=label)
    plt.xlabel("step")
    plt.ylabel(y_label)
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
