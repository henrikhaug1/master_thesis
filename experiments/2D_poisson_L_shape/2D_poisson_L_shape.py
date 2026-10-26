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
TITLE = "2D Laplace corner"
pi = jnp.pi

# ---------- Settings ----------
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
N_GRID = 64  # collocation grid is N_GRID x N_GRID minus the cut-out wedge
N_EVAL = 128  # error grid, likewise; it shares only the square's corners with N_GRID
BC_DENSITY = 75  # boundary points per unit length; about 600 points at 270 degrees
OMEGA_DEG = 200  # interior angle at the corner, 180 <= OMEGA_DEG <= 360; 270 is the L

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=1e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

# Widths give every model about as many parameters as the MLP.
MODELS = {
    "MLP": MLPConfig((2, 96, 96, 96, 1)),
    "KANN_cheb_decay": KANNConfig(
        (2, 36, 36, 36, 1),
        basis="cheb",
        basis_options=dict(degree=5, scale=2.0, decay=-2.0),
        input_options=dict(domain=((X_MIN, Y_MIN), (X_MAX, Y_MAX))),
    ),
}


# ---------- Problem ----------
def polar(x, y):
    """r and theta, with theta in [0, 2 pi) measured from the positive x-axis."""
    return jnp.sqrt(x**2 + y**2), jnp.mod(jnp.arctan2(y, x), 2 * pi)


def in_domain(x, y, omega):
    _, theta = polar(x, y)
    return theta <= omega + 1e-6  # tolerance keeps points on the theta = omega edge


def exact_solution(x, y, omega):
    lam = pi / omega
    r, theta = polar(x, y)
    return r**lam * jnp.sin(lam * theta)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return -laplacian(H)


def bc_fn(model, bc_pts, bc_vals):
    u = model(bc_pts)[:, 0]
    return jnp.mean((u - bc_vals) ** 2)


def segment(p0, p1, density, endpoint=True):
    """Points from p0 to p1, spaced by length so every edge has the same density."""
    p0, p1 = jnp.asarray(p0), jnp.asarray(p1)
    n = max(2, round(float(jnp.linalg.norm(p1 - p0)) * density))
    s = jnp.linspace(0.0, 1.0, n, endpoint=endpoint)[:, None]
    return (1.0 - s) * p0 + s * p1


def boundary_points(omega, density):
    """The outer square where theta <= omega, plus the two edges meeting at the corner."""
    corners = [
        (X_MAX, Y_MIN),
        (X_MAX, Y_MAX),
        (X_MIN, Y_MAX),
        (X_MIN, Y_MIN),
        (X_MAX, Y_MIN),
    ]
    # endpoint=False so each corner of the square appears once, not twice
    outer = jnp.concatenate(
        [segment(a, b, density, endpoint=False) for a, b in zip(corners, corners[1:])]
    )
    # Strict inequalities leave the points where the two corner edges meet the
    # square to those edges, so no point is counted twice.
    _, theta = polar(outer[:, 0], outer[:, 1])
    outer = outer[(theta > 1e-6) & (theta < omega - 1e-6)]

    # The theta = omega edge runs from where it meets the square back to the
    # corner, leaving out the corner itself, which edge_0 already contains.
    c, s = float(jnp.cos(omega)), float(jnp.sin(omega))
    exit_r = 1.0 / max(abs(c), abs(s))
    edge_0 = segment((0.0, 0.0), (X_MAX, 0.0), density)
    edge_omega = segment((exit_r * c, exit_r * s), (0.0, 0.0), density, endpoint=False)
    return jnp.concatenate([outer, edge_0, edge_omega])


def main():
    omega = OMEGA_DEG * pi / 180
    title = f"{TITLE} omega {OMEGA_DEG}"

    X_c, Y_c = jnp.meshgrid(
        jnp.linspace(X_MIN, X_MAX, N_GRID),
        jnp.linspace(Y_MIN, Y_MAX, N_GRID),
        indexing="ij",
    )
    inside = in_domain(X_c, Y_c, omega)
    pts = jnp.stack([X_c[inside], Y_c[inside]], axis=-1)

    xg = jnp.linspace(X_MIN, X_MAX, N_EVAL)
    yg = jnp.linspace(Y_MIN, Y_MAX, N_EVAL)
    X, Y = jnp.meshgrid(xg, yg, indexing="ij")
    mask = in_domain(X, Y, omega)
    eval_pts = jnp.stack([X[mask], Y[mask]], axis=-1)

    bc_pts = boundary_points(omega, BC_DENSITY)
    bc_vals = exact_solution(bc_pts[:, 0], bc_pts[:, 1], omega)
    print(
        f"omega = {OMEGA_DEG} deg, lambda = {float(pi / omega):.3f}, "
        f"{pts.shape[0]} interior and {bc_pts.shape[0]} boundary points"
    )

    results, run_dir = compare_models(
        MODELS,
        TRAIN,
        loss=lambda model: loss_fn(
            model, pts, residual, bc_fn=lambda m: bc_fn(m, bc_pts, bc_vals)
        ),
        predict_fn=lambda model: model(eval_pts)[:, 0],
        u_exact=exact_solution(eval_pts[:, 0], eval_pts[:, 1], omega),
        out_dir=OUT_DIR,
        title=title,
        problem=dict(
            omega_deg=OMEGA_DEG, n_grid=N_GRID, n_eval=N_EVAL, bc_density=BC_DENSITY
        ),
        run_name=f"{TRAIN.name}-omega{OMEGA_DEG}",
    )

    # ---------- Plotting ----------
    # Predictions only exist inside the domain; the cut-out wedge stays NaN.
    fields = {
        name: jnp.full(X.shape, jnp.nan).at[mask].set(median_prediction(runs))
        for name, runs in results.items()
    }
    U_exact = jnp.where(mask, exact_solution(X, Y, omega), jnp.nan)
    figs = run_dir / "figs"

    # y = 0.51 lies inside the domain for every omega of at least 180 degrees.
    j = 3 * N_EVAL // 4
    plot_solution_grid(
        xg,
        U_exact[:, j],
        {n: U[:, j] for n, U in fields.items()},
        "x",
        f"u(x, y = {float(yg[j]):.2f})",
        f"{title} at y = {float(yg[j]):.2f}",
        str(figs / f"{title.replace(' ', '_')}_slice.pdf"),
    )

    plot_fields(X, Y, U_exact, fields, "x", "y", title, figs, cbar_label="u(x,y)")


if __name__ == "__main__":
    main()
