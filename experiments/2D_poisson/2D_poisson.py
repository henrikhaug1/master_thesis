"""2D Poisson: -lap u = 2 pi^2 sin(pi x) sin(pi y) on the unit square, u = 0 on
the boundary. Exact: u = sin(pi x) sin(pi y)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from src.config import KANNConfig, MLPConfig, Stage, TrainConfig
from src.experiment import compare_models, median_prediction
from src.loss import loss_fn
from src.pinns import HardConstraint
from src.plotting import plot_fields, plot_solution_grid
from src.utils import derivatives, laplacian

OUT_DIR = Path(__file__).parent
TITLE = "2D Poisson"
pi = jnp.pi

# ---------- Settings ----------
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
N_GRID = 64  # collocation grid is N_GRID x N_GRID, also where the error is measured
N_BC = 100  # points per edge, used only when HARD_BC is False
HARD_BC = True  # impose u = 0 on the boundary exactly instead of as a loss term

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=3e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

CHEB = dict(degree=5, scale=2.0)
INPUT = dict(domain=((X_MIN, Y_MIN), (X_MAX, Y_MAX)))
MODELS = {
    "MLP": MLPConfig((2, 96, 96, 96, 1)),
    "KANN_spline_same_params": KANNConfig(
        (2, 32, 32, 32, 1),
        basis="bspline",
        input_options=dict(grid_range=(X_MIN, X_MAX)),
    ),
    "KANN_cheb": KANNConfig(
        (2, 32, 32, 32, 1), basis="cheb", basis_options=CHEB, input_options=INPUT
    ),
    "KANN_cheb_decay": KANNConfig(
        (2, 32, 32, 32, 1),
        basis="cheb",
        basis_options={**CHEB, "decay": -2.0},
        input_options=INPUT,
    ),
}


# ---------- Problem ----------
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


def hard_bc(net):
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

    return HardConstraint(net, phi=phi)


def main():
    xg = jnp.linspace(X_MIN, X_MAX, N_GRID)
    yg = jnp.linspace(Y_MIN, Y_MAX, N_GRID)
    X, Y = jnp.meshgrid(xg, yg, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # The four edges, used only when the BC is imposed softly.
    xb = jnp.linspace(X_MIN, X_MAX, N_BC)
    yb = jnp.linspace(Y_MIN, Y_MAX, N_BC)
    bc_pts = jnp.concatenate(
        [
            jnp.stack([xb, jnp.full_like(xb, Y_MIN)], axis=-1),
            jnp.stack([xb, jnp.full_like(xb, Y_MAX)], axis=-1),
            jnp.stack([jnp.full_like(yb, X_MIN), yb], axis=-1),
            jnp.stack([jnp.full_like(yb, X_MAX), yb], axis=-1),
        ]
    )

    if HARD_BC:
        loss = lambda model: loss_fn(model, pts, residual, hard_constraints=True)
    else:
        loss = lambda model: loss_fn(
            model, pts, residual, bc_fn=lambda m: bc_fn(m, bc_pts)
        )

    U_exact = exact_solution(X, Y)
    results, run_dir = compare_models(
        MODELS,
        TRAIN,
        loss,
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=U_exact.ravel(),
        out_dir=OUT_DIR,
        title=TITLE,
        problem=dict(
            domain=((X_MIN, Y_MIN), (X_MAX, Y_MAX)),
            n_grid=N_GRID,
            n_bc=N_BC,
            hard_bc=HARD_BC,
        ),
        wrap=hard_bc if HARD_BC else None,
    )

    # ---------- Plotting ----------
    fields = {
        name: median_prediction(runs).reshape(X.shape) for name, runs in results.items()
    }
    figs = run_dir / "figs"

    # Slice through the middle of the domain, u(x, y = 0.5).
    j = N_GRID // 2
    plot_solution_grid(
        xg,
        U_exact[:, j],
        {n: U[:, j] for n, U in fields.items()},
        "x",
        f"u(x, y = {float(yg[j]):.2f})",
        f"{TITLE} at y = {float(yg[j]):.2f}",
        str(figs / f"{TITLE.replace(' ', '_')}_slice.pdf"),
    )

    plot_fields(X, Y, U_exact, fields, "x", "y", TITLE, figs, cbar_label="u(x,y)")


if __name__ == "__main__":
    main()
