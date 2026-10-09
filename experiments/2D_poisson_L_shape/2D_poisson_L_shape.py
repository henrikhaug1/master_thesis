"""2D Laplace on the L-shape (-1, 1)^2 minus the quadrant x > 0, y < 0, with the
exact corner-singular solution u = r^(2/3) sin(2 theta / 3) as boundary data."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from src.config import KANNConfig, MLPConfig, Stage, TrainConfig
from src.experiment import compare_models, median_prediction
from src.loss import loss_fn
from src.plotting import plot_fields, plot_solution_grid
from src.utils import derivatives, laplacian

OUT_DIR = Path(__file__).parent
TITLE = "2D Poisson L-shape"
pi = jnp.pi

# ---------- Settings ----------
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
N_GRID = 64  # collocation grid is N_GRID x N_GRID minus the cut-out quadrant
N_BC = 100  # points per boundary edge

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=1e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

MODELS = {
    "MLP": MLPConfig((2, 96, 96, 96, 1)),
    "KANN_cheb_decay": KANNConfig(
        (2, 32, 32, 32, 1),
        basis="cheb",
        basis_options=dict(degree=5, scale=2.0, decay=-2.0),
        input_options=dict(domain=((X_MIN, Y_MIN), (X_MAX, Y_MAX))),
    ),
}


# ---------- Problem ----------
def in_domain(x, y):
    return ~((x > 0.0) & (y < 0.0))


def exact_solution(x, y):
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.mod(jnp.arctan2(y, x), 2 * pi)
    return r ** (2.0 / 3.0) * jnp.sin(2.0 * theta / 3.0)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return -laplacian(H)


def bc_fn(model, bc_pts, bc_vals):
    u = model(bc_pts)[:, 0]
    return jnp.mean((u - bc_vals) ** 2)


def boundary_points(n):
    """n points on each of the six edges of the L."""
    s = jnp.linspace(0.0, 1.0, n)
    edges = [
        (X_MIN + (X_MAX - X_MIN) * s, jnp.full_like(s, Y_MAX)),
        (jnp.full_like(s, X_MIN), Y_MIN + (Y_MAX - Y_MIN) * s),
        (X_MIN * (1.0 - s), jnp.full_like(s, Y_MIN)),
        (jnp.zeros_like(s), Y_MIN * (1.0 - s)),
        (X_MAX * s, jnp.zeros_like(s)),
        (jnp.full_like(s, X_MAX), Y_MAX * s),
    ]
    return jnp.concatenate([jnp.stack([x, y], axis=-1) for x, y in edges])


def main():
    xg = jnp.linspace(X_MIN, X_MAX, N_GRID)
    yg = jnp.linspace(Y_MIN, Y_MAX, N_GRID)
    X, Y = jnp.meshgrid(xg, yg, indexing="ij")
    mask = in_domain(X, Y)
    pts = jnp.stack([X[mask], Y[mask]], axis=-1)

    bc_pts = boundary_points(N_BC)
    bc_vals = exact_solution(bc_pts[:, 0], bc_pts[:, 1])

    results, run_dir = compare_models(
        MODELS,
        TRAIN,
        loss=lambda model: loss_fn(
            model, pts, residual, bc_fn=lambda m: bc_fn(m, bc_pts, bc_vals)
        ),
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=exact_solution(pts[:, 0], pts[:, 1]),
        out_dir=OUT_DIR,
        title=TITLE,
        problem=dict(n_grid=N_GRID, n_bc=N_BC),
    )

    # ---------- Plotting ----------
    # Predictions only exist inside the L; the cut-out quadrant stays NaN.
    fields = {
        name: jnp.full(X.shape, jnp.nan).at[mask].set(median_prediction(runs))
        for name, runs in results.items()
    }
    U_exact = jnp.where(mask, exact_solution(X, Y), jnp.nan)
    figs = run_dir / "figs"

    j = 3 * N_GRID // 4
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
