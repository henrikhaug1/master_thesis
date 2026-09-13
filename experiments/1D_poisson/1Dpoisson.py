import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import flax.nnx as nnx

from src.pinns import MLP, KANN, HardConstraint
from src.loss import loss_fn
from src.utils import derivatives, laplacian
from src.sweep import compare_models
from src.bases import BSplineBasis, ChebyshevBasis


pi = jnp.pi
X_MIN, X_MAX = 0.0, 1.0
OUT_DIR = Path(__file__).parent
HARD_BC = True


def exact_solution(x):
    return jnp.sin(pi * x)


def f(x):
    return (pi**2) * jnp.sin(pi * x)


def residual(model, pts):
    u, g, H = derivatives(model, pts, order=2)
    return -laplacian(H) - f(pts[:, 0])


def bc_fn(model, bc_pts=jnp.array([[0.0], [1.0]])):
    u = model(bc_pts)[:, 0]
    return jnp.mean(u**2)


def hard_bc(model_fn):
    """u = 4x(1-x) * N(x), so u(0) = u(1) = 0 exactly."""
    return lambda rngs: HardConstraint(
        model_fn(rngs), phi=lambda x: 4.0 * x * (1.0 - x)
    )


def main():
    x = jnp.linspace(X_MIN, X_MAX, 200)
    pts = x[:, None]
    analytical_solution = exact_solution(x)

    models = {
        "MLP": lambda rngs: MLP([1, 48, 48, 48, 1], act_fun=nnx.silu, rngs=rngs),
        "KANN_spline": lambda rngs: KANN(
            [1, 16, 16, 16, 1],
            basis_fn=BSplineBasis,
            rngs=rngs,
        ),
        "KANN_cheb": lambda rngs: KANN(
            [1, 16, 16, 16, 1],
            basis_fn=lambda: ChebyshevBasis(degree=5, scale=2.0),
            input_basis_fn=lambda: ChebyshevBasis(degree=5, domain=(X_MIN, X_MAX)),
            rngs=rngs,
        ),
    }

    if HARD_BC:
        models = {name: hard_bc(fn) for name, fn in models.items()}
        loss = lambda model: loss_fn(model, pts, residual, hard_constraints=True)
    else:
        loss = lambda model: loss_fn(model, pts, residual, bc_fn=bc_fn)

    compare_models(
        models=models,
        loss=loss,
        predict_fn=lambda model: model(x[:, None])[:, 0],
        u_exact=analytical_solution,
        x=x,
        seeds=(0, 1, 2),
        out_dir=OUT_DIR,
        title="1D Poisson",
        x_label="x",
        y_label="u(x)",
        steps=5000,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()
