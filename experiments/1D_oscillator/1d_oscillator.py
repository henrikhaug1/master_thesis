import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import flax.nnx as nnx

from src.pinns import MLP, KANN
from src.bases import ChebyshevBasis
from src.loss import loss_fn
from src.utils import derivatives
from src.sweep import compare_models


m = 1.0
mu = 0.4
k = 4.0

OUT_DIR = Path(__file__).parent


def exact_solution(t):
    w = jnp.sqrt(4 * k - mu**2) / 2
    return jnp.exp(-mu / 2 * t) * jnp.cos(w * t)


def residual(model, pts, m=m, mu=mu, k=k):
    u, g, H = derivatives(model, pts, order=2)
    return m * H[:, 0, 0] + mu * g[:, 0] + k * u


def ic_fn(model, t0=jnp.array([[0.0]]), u0=1.0, v0=0.0):
    u, g = derivatives(model, t0, order=1)
    return jnp.mean((u - u0) ** 2) + jnp.mean((g[:, 0] - v0) ** 2)


T_MIN, T_MAX = 0.0, 10.0


def main():
    t = jnp.linspace(T_MIN, T_MAX, 200)
    pts = t[:, None]

    # --------- Analytical ----------
    analytical_solution = exact_solution(t)

    # ---------- Sweep ----------
    compare_models(
        models={
            "MLP": lambda rngs: MLP([1, 48, 48, 48, 1], act_fun=nnx.silu, rngs=rngs),
            "KANN_spline_same_width": lambda rngs: KANN([1, 48, 48, 48, 1], rngs=rngs),
            "KANN_spline_same_params": lambda rngs: KANN([1, 16, 16, 16, 1], rngs=rngs),
            "KANN_cheb": lambda rngs: KANN(
                [1, 16, 16, 16, 1],
                basis_fn=lambda: ChebyshevBasis(degree=5, scale=2.0),
                input_basis_fn=lambda: ChebyshevBasis(degree=5, domain=(T_MIN, T_MAX)),
                rngs=rngs,
            ),
        },
        loss=lambda model: loss_fn(model, pts, residual, ic_fn),
        predict_fn=lambda model: model(pts)[:, 0],
        u_exact=analytical_solution,
        x=t,
        seeds=(0, 1, 2),
        out_dir=OUT_DIR,
        title="1D Oscillator",
        x_label="t",
        y_label="u(t)",
        steps=5000,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()
