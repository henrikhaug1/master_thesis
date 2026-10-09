"""Damped oscillator: m u'' + mu u' + k u = 0 on [0, 10], u(0) = 1, u'(0) = 0."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from src.config import KANNConfig, MLPConfig, Stage, TrainConfig
from src.experiment import compare_models
from src.loss import loss_fn
from src.utils import derivatives

OUT_DIR = Path(__file__).parent
TITLE = "1D Oscillator"

# ---------- Settings ----------
M, MU, K = 1.0, 0.4, 4.0  # mass, damping, stiffness
T_MIN, T_MAX = 0.0, 10.0
N_POINTS = 200  # collocation points, also where the error is measured

TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=1e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)

CHEB = dict(degree=5, scale=2.0)
INPUT = dict(domain=(T_MIN, T_MAX))
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
def exact_solution(t):
    w = jnp.sqrt(4 * K - MU**2) / 2
    return jnp.exp(-MU / 2 * t) * jnp.cos(w * t)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return M * H[:, 0, 0] + MU * g[:, 0] + K * u


def ic_fn(model, t0=jnp.array([[T_MIN]]), u0=1.0, v0=0.0):
    u, g = derivatives(model, t0, order=1)
    return jnp.mean((u - u0) ** 2) + jnp.mean((g[:, 0] - v0) ** 2)


def main():
    t = jnp.linspace(T_MIN, T_MAX, N_POINTS)
    pts = t[:, None]

    compare_models(
        MODELS,
        TRAIN,
        loss=lambda model: loss_fn(model, pts, residual, ic_fn),
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=exact_solution(t),
        out_dir=OUT_DIR,
        title=TITLE,
        problem=dict(m=M, mu=MU, k=K, domain=(T_MIN, T_MAX), n_points=N_POINTS),
        x=t,
        x_label="t",
        y_label="u(t)",
    )


if __name__ == "__main__":
    main()
