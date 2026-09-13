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


pi = jnp.pi
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
OUT_DIR = Path(__file__).parent
HARD_BC = True


def exact_solution(x, y):
    return jnp.sin(pi * x) * jnp.sin(pi * y)


def f(x, y):
    return 2 * (pi**2) * jnp.sin(pi * x) * jnp.sin(pi * y)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return -laplacian(H) - f(pts[:, 0], pts[:, 1])


def bc_fn(model, bc_pts):
    u = model(bc_pts)[:, 0]
    return jnp.mean(u**2)


def hard_bc(model_fn):
    """u = phi(x,y) * N(x,y) with phi = 0 on all four edges, so u = 0 there exactly."""
    lx, ly = X_MAX - X_MIN, Y_MAX - Y_MIN

    def phi(pts):
        x, y = pts[:, :1], pts[:, 1:2]
        # 16 x(1-x) y(1-y) on the unit square; the normalisation puts max|phi| = 1.
        return (
            16.0
            * (x - X_MIN)
            * (X_MAX - x)
            * (y - Y_MIN)
            * (Y_MAX - y)
            / (lx**2 * ly**2)
        )

    return lambda rngs: HardConstraint(model_fn(rngs), phi=phi)


def main():
    xg = jnp.linspace(X_MIN, X_MAX, 64)
    yg = jnp.linspace(Y_MIN, Y_MAX, 64)
    X, Y = jnp.meshgrid(xg, yg, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (4096, 2)

    # The four edges, used only when the BC is imposed softly.
    xb = jnp.linspace(X_MIN, X_MAX, 100)
    yb = jnp.linspace(Y_MIN, Y_MAX, 100)
    bc_pts = jnp.concatenate(
        [
            jnp.stack([xb, jnp.full_like(xb, Y_MIN)], axis=-1),
            jnp.stack([xb, jnp.full_like(xb, Y_MAX)], axis=-1),
            jnp.stack([jnp.full_like(yb, X_MIN), yb], axis=-1),
            jnp.stack([jnp.full_like(yb, X_MAX), yb], axis=-1),
        ]
    )

    U_exact = exact_solution(X, Y)

    models = {
        "MLP": lambda rngs: MLP([2, 96, 96, 96, 1], act_fun=nnx.silu, rngs=rngs),
        "KANN_spline": lambda rngs: KANN(
            [2, 32, 32, 32, 1],
            basis_fn=BSplineBasis,
            # The input range is known exactly here, unlike the hidden layers'.
            input_basis_fn=lambda: BSplineBasis(grid_range=(X_MIN, X_MAX)),
            rngs=rngs,
        ),
        "KANN_cheb": lambda rngs: KANN(
            [2, 32, 32, 32, 1],
            basis_fn=lambda: ChebyshevBasis(degree=5, scale=2.0),
            input_basis_fn=lambda: ChebyshevBasis(
                degree=5,
                domain=(jnp.array([X_MIN, Y_MIN]), jnp.array([X_MAX, Y_MAX])),
            ),
            rngs=rngs,
        ),
    }

    if HARD_BC:
        models = {name: hard_bc(fn) for name, fn in models.items()}
        loss = lambda model: loss_fn(model, pts, residual, hard_constraints=True)
    else:
        loss = lambda model: loss_fn(
            model, pts, residual, bc_fn=lambda m: bc_fn(m, bc_pts)
        )

    results = compare_models(
        models=models,
        loss=loss,
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=U_exact.ravel(),
        x=None,
        seeds=(0, 1, 2),
        out_dir=OUT_DIR,
        title="2D Poisson",
        steps=5000,
        lr=1e-3,
    )

    # ---------- Plotting ----------
    # Median prediction per model, reshaped back onto the (x, y) grid.
    fields = {
        name: np.median(np.stack([r["u"] for r in runs]), axis=0).reshape(X.shape)
        for name, runs in results.items()
    }
    figs = OUT_DIR / "figs"

    # Slice through the middle of the domain, u(x, y = 0.5).
    j = len(yg) // 2
    plot_solutions(
        xg,
        {"exact": U_exact[:, j], **{n: U[:, j] for n, U in fields.items()}},
        "x",
        f"u(x, y = {float(yg[j]):.2f})",
        f"2D Poisson at y = {float(yg[j]):.2f}",
        str(figs / "2D_poisson_slice.pdf"),
    )

    # Full fields and their error against the exact solution.
    for name, U in [("exact", U_exact), *fields.items()]:
        plot_field(
            X,
            Y,
            U,
            "x",
            "y",
            f"2D Poisson - {name}",
            str(figs / f"2D_poisson_field_{name}.pdf"),
            cbar_label="u(x,y)",
        )

    for name, U in fields.items():
        plot_field(
            X,
            Y,
            jnp.abs(U - U_exact),
            "x",
            "y",
            f"2D Poisson - |{name} - exact|",
            str(figs / f"2D_poisson_error_{name}.pdf"),
            cmap="magma",
            cbar_label="abs. error",
        )


if __name__ == "__main__":
    main()
