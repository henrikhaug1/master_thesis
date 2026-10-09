"""1D wave equation: u_tt = c^2 u_xx on [0, L] x [0, T], u = 0 at x = 0, L,
u(x, 0) = sin(kx), u_t(x, 0) = 0. Exact: u = sin(kx) cos(ckt)."""

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
from src.utils import derivatives

OUT_DIR = Path(__file__).parent
TITLE = "1D Wave Equation"

# ---------- Settings ----------
C = 1.0  # wave speed
L = 2 * jnp.pi
K = jnp.pi / L  # wave number of the initial condition
T = 2 * L / C  # one full period of cos(c k t)
N_GRID = 60  # collocation grid is N_GRID x N_GRID, also where the error is measured
N_IC = 100  # points on t = 0
N_BC = 100  # points on each of x = 0 and x = L, used only when HARD_BC is False
HARD_BC = True  # impose u = 0 at x = 0, L exactly instead of as a loss term
W_BC = 2.0  # weight of the boundary loss when HARD_BC is False

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=3e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

CHEB = dict(degree=5, scale=2.0)
INPUT = dict(domain=((0.0, 0.0), (L, T)))
MODELS = {
    "MLP": MLPConfig((2, 96, 96, 96, 1)),
    # Width cut until the parameter count matches the MLP instead.
    "KANN_spline_same_params": KANNConfig(
        (2, 32, 32, 32, 1),
        basis="bspline",
        basis_options=dict(grid_range=(-0.5, 13.0)),
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
def exact_solution(x, t):
    return jnp.sin(K * x) * jnp.cos(C * K * t)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return H[:, 1, 1] - (C**2 * H[:, 0, 0])


def bc_fn(model, bc_pts):
    u = model(bc_pts)[:, 0]
    return jnp.mean(u**2)


def ic_fn(model, ic_pts):
    u, g = derivatives(model, ic_pts, order=1)
    return jnp.mean((u - jnp.sin(K * ic_pts[:, 0])) ** 2) + jnp.mean(g[:, 1] ** 2)


def hard_bc(net):
    """u = 4x(L-x)/L^2 * N(x,t), so u(0,t) = u(L,t) = 0 exactly."""
    return HardConstraint(net, phi=lambda xt: 4.0 * xt[:, :1] * (L - xt[:, :1]) / L**2)


def main():
    xg = jnp.linspace(0, L, N_GRID)
    tg = jnp.linspace(0, T, N_GRID)
    X, T_grid = jnp.meshgrid(xg, tg, indexing="ij")
    pts = jnp.stack([X.ravel(), T_grid.ravel()], axis=-1)

    x_ic = jnp.linspace(0, L, N_IC)  # bottom edge t = 0
    ic_pts = jnp.stack([x_ic, jnp.zeros_like(x_ic)], axis=-1)

    t_bc = jnp.linspace(0, T, N_BC)  # side edges x = 0, L
    bc_pts = jnp.concatenate(
        [
            jnp.stack([jnp.zeros_like(t_bc), t_bc], axis=-1),
            jnp.stack([jnp.full_like(t_bc, L), t_bc], axis=-1),
        ]
    )

    if HARD_BC:
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
            w_bc=W_BC,
        )

    U_exact = exact_solution(X, T_grid)
    results, run_dir = compare_models(
        MODELS,
        TRAIN,
        loss,
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=U_exact.ravel(),
        out_dir=OUT_DIR,
        title=TITLE,
        problem=dict(
            c=C,
            L=L,
            T=T,
            n_grid=N_GRID,
            n_ic=N_IC,
            n_bc=N_BC,
            hard_bc=HARD_BC,
            w_bc=W_BC,
        ),
        wrap=hard_bc if HARD_BC else None,
    )

    # ---------- Plotting ----------
    fields = {
        name: median_prediction(runs).reshape(X.shape) for name, runs in results.items()
    }
    figs = run_dir / "figs"

    # Snapshots u(x, .) at a few fixed times.
    for j in (0, N_GRID // 4, N_GRID // 2):
        plot_solution_grid(
            xg,
            U_exact[:, j],
            {n: U[:, j] for n, U in fields.items()},
            "x",
            "u(x,t)",
            f"{TITLE} at t = {float(tg[j]):.2f}",
            str(figs / f"{TITLE.replace(' ', '_')}_snapshot_t{j}.pdf"),
        )

    plot_fields(X, T_grid, U_exact, fields, "x", "t", TITLE, figs, cbar_label="u(x,t)")


if __name__ == "__main__":
    main()
