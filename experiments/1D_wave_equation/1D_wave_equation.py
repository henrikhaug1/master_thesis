import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from src.pinns import MLP, KANN
from src.loss import loss_fn
from src.utils import partials
from src.plotting import plot_solutions, plot_field
from src.sweep import compare_models
from src.bases import BSplineBasis


c = 1.0
L = 2 * jnp.pi
k = jnp.pi / L
T = 2 * L / c  # one full period of cos(c k t)

OUT_DIR = Path(__file__).parent


def exact_solution(x, t):
    return jnp.sin(k * x) * jnp.cos(c * k * t)


def residual(model, x, t):
    u, u_x, u_t, u_xx, u_tt = partials(model, x, t)
    return u_tt - (c**2 * u_xx)


def bc_fn(model, t):
    x0 = jnp.zeros_like(t)
    xL = jnp.full_like(t, L)
    u_left = model(jnp.stack([x0, t], axis=-1))[:, 0]
    u_right = model(jnp.stack([xL, t], axis=-1))[:, 0]
    return jnp.mean(u_left**2) + jnp.mean(u_right**2)


def ic_fn(model, x):
    t0 = jnp.zeros_like(x)
    u, u_x, u_t, u_xx, u_tt = partials(model, x, t0)
    return jnp.mean((u - jnp.sin(k * x)) ** 2) + jnp.mean(u_t**2)


def main():
    xg = jnp.linspace(0, L, 60)
    tg = jnp.linspace(0, T, 60)
    X, T_grid = jnp.meshgrid(xg, tg, indexing="ij")
    collocation = (X.ravel(), T_grid.ravel())

    x_ic = jnp.linspace(0, L, 100)  # bottom edge t = 0
    t_bc = jnp.linspace(0, T, 100)  # side edges x = 0, L

    XT = jnp.stack([X.ravel(), T_grid.ravel()], axis=-1)
    U_exact = exact_solution(X, T_grid)

    results = compare_models(
        models={
            "MLP": lambda rngs: MLP([2, 96, 96, 96, 1], act_fun=nnx.silu, rngs=rngs),
            "KANN": lambda rngs: KANN(
                [2, 32, 32, 32, 1],
                basis_fn=lambda: BSplineBasis(grid_range=(-0.5, 13.0)),
                rngs=rngs,
            ),
        },
        loss=lambda model: loss_fn(
            model,
            collocation,
            residual=residual,
            ic_fn=lambda m: ic_fn(m, x_ic),
            bc_fn=lambda m: bc_fn(m, t_bc),
        ),
        predict_fn=lambda model: model(XT)[:, 0],
        u_exact=U_exact.ravel(),
        x=None,
        seeds=(0, 1, 2),
        out_dir=OUT_DIR,
        title="1D Wave Equation",
        steps=5000,
        lr=1e-3,
    )

    # ---------- Plotting ----------
    # Median prediction per model, reshaped back onto the (x, t) grid.
    fields = {
        name: np.median(np.stack([r["u"] for r in runs]), axis=0).reshape(X.shape)
        for name, runs in results.items()
    }
    figs = OUT_DIR / "figs"

    # Snapshots u(x, .) at a few fixed times.
    for j in [0, len(tg) // 4, len(tg) // 2]:
        plot_solutions(
            xg,
            {"exact": U_exact[:, j], **{n: U[:, j] for n, U in fields.items()}},
            "x",
            "u(x,t)",
            f"1D Wave at t = {float(tg[j]):.2f}",
            str(figs / f"1D_wave_snapshot_t{j}.pdf"),
        )

    # Full space-time fields and their error against the exact solution.
    for name, U in [("exact", U_exact), *fields.items()]:
        plot_field(
            X,
            T_grid,
            U,
            "x",
            "t",
            f"1D Wave - {name}",
            str(figs / f"1D_wave_field_{name}.pdf"),
            cbar_label="u(x,t)",
        )

    for name, U in fields.items():
        plot_field(
            X,
            T_grid,
            jnp.abs(U - U_exact),
            "x",
            "t",
            f"1D Wave - |{name} - exact|",
            str(figs / f"1D_wave_error_{name}.pdf"),
            cmap="magma",
            cbar_label="abs. error",
        )


if __name__ == "__main__":
    main()
