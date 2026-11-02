"""2D Poisson: -lap u = 2 pi^2 sin(pi x) sin(pi y) on the unit square, u = 0 on
the boundary. Exact: u = sin(pi x) sin(pi y)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from src.config import GridCollocation, KANNConfig, MLPConfig, Stage, TrainConfig
from src.experiment import compare_models, median_prediction
from src.loss import dirichlet_loss, loss_fn
from src.pinns import HardConstraint
from src.plotting import plot_fields, plot_solution_grid
from src.sampling import Domain, segment_points
from src.utils import derivatives, laplacian

OUT_DIR = Path(__file__).parent
TITLE = "2D Poisson"
pi = jnp.pi

# ---------- Settings ----------
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
COLLOCATION = GridCollocation(n=64)  # interior points; see src.config for others
N_EVAL = 128  # error grid is N_EVAL x N_EVAL; it shares only the corners with the grid
N_BC = 100  # points per edge, used only when HARD_BC is False
HARD_BC = True  # impose u = 0 on the boundary exactly instead of as a loss term

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=3e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

CHEB = dict(degree=5, scale=2.0)
INPUT = dict(domain=((X_MIN, Y_MIN), (X_MAX, Y_MAX)))
# Widths give every model about as many parameters as the MLP.
MODELS = {
    "MLP": MLPConfig((2, 96, 96, 96, 1)),
    "KANN_spline_same_params": KANNConfig(
        (2, 32, 32, 32, 1),
        basis="bspline",
        basis_options=dict(scale=2.0),
        input_options=INPUT,
    ),
    "KANN_cheb": KANNConfig(
        (2, 36, 36, 36, 1), basis="cheb", basis_options=CHEB, input_options=INPUT
    ),
    "KANN_cheb_decay": KANNConfig(
        (2, 36, 36, 36, 1),
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
    pts, weights = COLLOCATION.points(Domain((X_MIN, Y_MIN), (X_MAX, Y_MAX)))

    xg = jnp.linspace(X_MIN, X_MAX, N_EVAL)
    yg = jnp.linspace(Y_MIN, Y_MAX, N_EVAL)
    X, Y = jnp.meshgrid(xg, yg, indexing="ij")
    eval_pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # The four edges, used only when the BC is imposed softly.
    bc_pts = jnp.concatenate(
        [
            segment_points((X_MIN, Y_MIN), (X_MAX, Y_MIN), N_BC),
            segment_points((X_MIN, Y_MAX), (X_MAX, Y_MAX), N_BC),
            segment_points((X_MIN, Y_MIN), (X_MIN, Y_MAX), N_BC),
            segment_points((X_MAX, Y_MIN), (X_MAX, Y_MAX), N_BC),
        ]
    )

    def loss(model):
        if HARD_BC:
            return loss_fn(model, pts, residual, hard_constraints=True, weights=weights)
        return loss_fn(
            model,
            pts,
            residual,
            bc_fn=lambda m: dirichlet_loss(m, bc_pts),
            weights=weights,
        )

    U_exact = exact_solution(X, Y)
    results, run_dir = compare_models(
        MODELS,
        TRAIN,
        loss,
        predict_fn=lambda model: model(eval_pts)[:, 0],
        u_exact=U_exact.ravel(),
        out_dir=OUT_DIR,
        title=TITLE,
        problem=dict(
            domain=((X_MIN, Y_MIN), (X_MAX, Y_MAX)),
            n_eval=N_EVAL,
            n_bc=N_BC,
            hard_bc=HARD_BC,
        ),
        collocation=COLLOCATION,
        wrap=hard_bc if HARD_BC else None,
    )

    # ---------- Plotting ----------
    fields = {
        name: median_prediction(runs).reshape(X.shape) for name, runs in results.items()
    }
    figs = run_dir / "figs"

    # Slice through the middle of the domain, u(x, y = 0.5).
    j = N_EVAL // 2
    plot_solution_grid(
        xg,
        U_exact[:, j],
        {n: U[:, j] for n, U in fields.items()},
        "x",
        f"u(x, y = {float(yg[j]):.2f})",
        f"{TITLE} at y = {float(yg[j]):.2f}",
        str(figs / f"{TITLE.replace(' ', '_')}_slice.pdf"),
    )

    plot_fields(
        X,
        Y,
        U_exact,
        fields,
        "x",
        "y",
        TITLE,
        figs,
        cbar_label="u(x,y)",
        aspect="equal",
    )


if __name__ == "__main__":
    main()
