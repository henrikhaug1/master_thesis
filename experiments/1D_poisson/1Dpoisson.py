import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.pinns import MLP, KANN
from src.loss import loss_fn
from src.utils import derivatives
from src.train import train
from src.plotting import plot_solutions, plot_losses


pi = jnp.pi
key = jax.random.PRNGKey(0)


def f(x):
    return (pi**2) * jnp.sin(pi * x)


def residual(model, x):
    u, u_x, u_xx = derivatives(model, t=x, order=2)
    return -u_xx - f(x)


def bc_fn(model, x0=jnp.array([0.0]), x1=jnp.array([1.0])):
    u0 = derivatives(model, x0, order=0)[0]
    u1 = derivatives(model, x1, order=0)[0]
    return jnp.mean(u0**2) + jnp.mean(u1**2)


def main():
    x = jnp.linspace(0, 1, 200)
    # ---------- MLP ----------
    MLP_model = MLP(layer_sizes=[1, 16, 16, 16, 1], act_fun=nnx.silu, rngs=nnx.Rngs(0))
    MLP_history = train(
        model=MLP_model,
        loss=lambda model: loss_fn(model, x, residual=residual, ic_fn=bc_fn),
        steps=5000,
        lr=0.001,
    )
    u_MLP = MLP_model(x[:, None])[:, 0]
    # ---------- KANN ----------
    KANN_model = KANN(layer_sizes=[1, 16, 16, 16, 1], rngs=nnx.Rngs(0))
    KANN_history = train(
        model=KANN_model,
        loss=lambda model: loss_fn(model, x, residual=residual, ic_fn=bc_fn),
        steps=5000,
        lr=0.001,
    )
    u_KANN = KANN_model(x[:, None])[:, 0]
    # ---------- Plotting ----------
    dict_sol = {"analytical": jnp.sin(pi * x), "MLP": u_MLP, "KANN": u_KANN}
    dict_hist = {"MLP": MLP_history, "KANN": KANN_history}
    plot_solutions(
        x,
        dict_sol,
        "x",
        "u(x)",
        "Poisson Equation",
        "experiments/1D_poisson/figs/1D_poisson_MLP_VS_KANN.pdf",
    )
    plot_losses(
        dict_hist, "experiments/1D_poisson/figs/loss_history.pdf", "Loss 1D Poisson"
    )


if __name__ == "__main__":
    main()
