import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from src.pinns import MLP, KANN, HardConstraint
from src.loss import loss_fn
from src.utils import derivatives, laplacian
from src.plotting import plot_solutions, plot_field
from src.sweep import compare_models
from src.bases import BSplineBasis, ChebyshevBasis


c = 1.0
L = 2 * jnp.pi
k = jnp.pi / L
T = 2 * L / c  # one full period of cos(c k t)

OUT_DIR = Path(__file__).parent

HARD_BC = True


def exact_solution(x, t):
    return jnp.sin(k * x) * jnp.cos(c * k * t)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return H[:, 1, 1] - (c**2 * H[:, 0, 0])


def bc_fn(model, bc_pts):
    u = model(bc_pts)[:, 0]
    return jnp.mean(u**2)


def ic_fn(model, ic_pts):
    u, g = derivatives(model, ic_pts, order=1)
    return jnp.mean((u - jnp.sin(k * ic_pts[:, 0])) ** 2) + jnp.mean(g[:, 1] ** 2)


def hard_bc(model_fn):
    """u = 4x(L-x)/L^2 * N(x,t), so u(0,t) = u(L,t) = 0 exactly."""
    return lambda rngs: HardConstraint(
        model_fn(rngs), phi=lambda xt: 4.0 * xt[:, :1] * (L - xt[:, :1]) / L**2
    )


def main():
    xg = jnp.linspace(0, L, 60)
    tg = jnp.linspace(0, T, 60)
    X, T_grid = jnp.meshgrid(xg, tg, indexing="ij")
    pts = jnp.stack([X.ravel(), T_grid.ravel()], axis=-1)

    x_ic = jnp.linspace(0, L, 100)  # bottom edge t = 0
    ic_pts = jnp.stack([x_ic, jnp.zeros_like(x_ic)], axis=-1)

    t_bc = jnp.linspace(0, T, 100)  # side edges x = 0, L
    bc_pts = jnp.concatenate(
        [
            jnp.stack([jnp.zeros_like(t_bc), t_bc], axis=-1),
            jnp.stack([jnp.full_like(t_bc, L), t_bc], axis=-1),
        ]
    )

    U_exact = exact_solution(X, T_grid)

    models = {
        "MLP": lambda rngs: MLP([2, 96, 96, 96, 1], act_fun=nnx.silu, rngs=rngs),
        # Same layout as the MLP -- every edge carries n_basis coefficients, so
        # this costs roughly 9x the MLP's parameters (and trains far slower).
        "KANN_spline_same_width": lambda rngs: KANN(
            [2, 96, 96, 96, 1],
            basis_fn=lambda: BSplineBasis(grid_range=(-0.5, 13.0)),
            rngs=rngs,
        ),
        # Width cut until the parameter count matches the MLP instead.
        "KANN_spline_same_params": lambda rngs: KANN(
            [2, 32, 32, 32, 1],
            basis_fn=lambda: BSplineBasis(grid_range=(-0.5, 13.0)),
            rngs=rngs,
        ),
        "KANN_cheb": lambda rngs: KANN(
            [2, 32, 32, 32, 1],
            basis_fn=lambda: ChebyshevBasis(degree=5, scale=2.0),
            input_basis_fn=lambda: ChebyshevBasis(
                degree=5, domain=(jnp.zeros(2), jnp.array([L, T]))
            ),
            rngs=rngs,
        ),
    }

    if HARD_BC:
        models = {name: hard_bc(fn) for name, fn in models.items()}
        loss = lambda model: loss_fn(
            model, pts, residual=residual, ic_fn=lambda m: ic_fn(m, ic_pts)
        )
    else:
        loss = lambda model: loss_fn(
            model,
            pts,
            residual=residual,
            ic_fn=lambda m: ic_fn(m, ic_pts),
            bc_fn=lambda m: bc_fn(m, bc_pts),
            w_bc=2.0,
        )

    results = compare_models(
        models=models,
        loss=loss,
        predict_fn=lambda model: model(pts)[:, 0],
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
