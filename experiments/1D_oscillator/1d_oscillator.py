import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import flax.nnx as nnx

from src.pinns import MLP, KANN
from src.loss import loss_fn
from src.utils import derivatives
from src.train import train
from src.plotting import plot_solutions, plot_losses


m = 1.0
mu = 0.4
k = 4.0


def exact_solution(t):
    w = jnp.sqrt(4 * k - mu**2) / 2
    return jnp.exp(-mu / 2 * t) * jnp.cos(w * t)


def residual(model, t, m=m, mu=mu, k=k):
    u, u_t, u_tt = derivatives(model, t, order=2)
    return m * u_tt + mu * u_t + k * u


def ic_fn(model, t0=jnp.array([0.0]), u0=1.0, v0=0.0):
    u, u_t = derivatives(model, t0, order=1)
    return jnp.mean((u - u0) ** 2) + jnp.mean((u_t - v0) ** 2)


def main():
    t = jnp.linspace(0, 10, 200)

    # --------- Analytical ----------
    analytical_solution = exact_solution(t)
    # ---------- MLP ----------
    MLP_model = MLP(layer_sizes=[1, 16, 16, 16, 1], act_fun=nnx.silu, rngs=nnx.Rngs(0))
    train_history_MLP = train(
        MLP_model,
        lambda model: loss_fn(model, t, residual, ic_fn),
        steps=5000,
        lr=0.001,
    )
    u_MLP = MLP_model(t[:, None])[:, 0]

    # ----------KANN----------
    KANN_model = KANN(layer_sizes=[1, 16, 16, 16, 1], rngs=nnx.Rngs(0))
    train_history_KANN = train(
        KANN_model,
        lambda model: loss_fn(model, t, residual, ic_fn),
        steps=5000,
        lr=0.001,
    )
    u_KANN = KANN_model(t[:, None])[:, 0]

    # ----------Plotting---------
    solution_dict = {
        "Analytical": analytical_solution,
        "u_MLP": u_MLP,
        "u_KANN": u_KANN,
    }
    history_dict = {"MLP": train_history_MLP, "KANN": train_history_KANN}
    plot_solutions(
        t,
        solution_dict,
        "t",
        "u(t)",
        "1D Oscillating Spring",
        "experiments/1d_oscillator/figs/1D_oscillator_MLP_vs_KANN.pdf",
    )
    plot_losses(
        history_dict,
        "experiments/1d_oscillator/figs/1D_oscillator_MLP_vs_KANN_loss.pdf",
        "Loss Oscillating Spring",
    )


if __name__ == "__main__":
    main()
