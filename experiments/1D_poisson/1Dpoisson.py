"""1D Poisson: -u'' = pi^2 sin(pi x) on [0, 1], u(0) = u(1) = 0. Exact: u = sin(pi x)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from src.config import KANNConfig, MLPConfig, Stage, TrainConfig
from src.experiment import compare_models
from src.loss import loss_fn
from src.pinns import HardConstraint
from src.utils import derivatives, laplacian

OUT_DIR = Path(__file__).parent
TITLE = "1D Poisson"
pi = jnp.pi

# ---------- Settings ----------
X_MIN, X_MAX = 0.0, 1.0
N_POINTS = 200  # collocation points, also where the error is measured
HARD_BC = True  # impose u(0) = u(1) = 0 exactly instead of as a loss term

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=3e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

CHEB = dict(degree=5, scale=2.0)
INPUT = dict(domain=(X_MIN, X_MAX))
MODELS = {
    "MLP": MLPConfig((1, 48, 48, 48, 1)),
    "KANN_spline_same_params": KANNConfig((1, 16, 16, 16, 1), basis="bspline"),
    "KANN_cheb": KANNConfig(
        (1, 16, 16, 16, 1), basis="cheb", basis_options=CHEB, input_options=INPUT
    ),
    "KANN_cheb_decay": KANNConfig(
        (1, 16, 16, 16, 1),
        basis="cheb",
        basis_options={**CHEB, "decay": -2.0},
        input_options=INPUT,
    ),
}


# ---------- Problem ----------
def exact_solution(x):
    return jnp.sin(pi * x)


def f(x):
    return (pi**2) * jnp.sin(pi * x)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return -laplacian(H) - f(pts[:, 0])


def bc_fn(model, bc_pts=jnp.array([[X_MIN], [X_MAX]])):
    u = model(bc_pts)[:, 0]
    return jnp.mean(u**2)


def hard_bc(net):
    """u = 4x(1-x) * N(x), so u(0) = u(1) = 0 exactly."""
    return HardConstraint(net, phi=lambda x: 4.0 * x * (1.0 - x))


def main():
    x = jnp.linspace(X_MIN, X_MAX, N_POINTS)
    pts = x[:, None]

    if HARD_BC:
        loss = lambda model: loss_fn(model, pts, residual, hard_constraints=True)
    else:
        loss = lambda model: loss_fn(model, pts, residual, bc_fn=bc_fn)

    compare_models(
        MODELS,
        TRAIN,
        loss,
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=exact_solution(x),
        out_dir=OUT_DIR,
        title=TITLE,
        problem=dict(domain=(X_MIN, X_MAX), n_points=N_POINTS, hard_bc=HARD_BC),
        wrap=hard_bc if HARD_BC else None,
        x=x,
        x_label="x",
        y_label="u(x)",
    )


if __name__ == "__main__":
    main()
