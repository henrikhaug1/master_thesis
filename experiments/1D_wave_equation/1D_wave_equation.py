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
from src.utils import partials
from src.train import train
from src.plotting import plot_solutions, plot_losses, plot_field


c = 1.0
L = 2 * jnp.pi
k = jnp.pi / L


def exact_solution(x, t):
    return jnp.sin(k * x) * jnp.cos(c * k * t)


def residual(model, x, t):
    u, u_x, u_t, u_xx, u_tt = partials(model, x, t)
    return u_tt - (c**2 * u_xx)


def bc_fn(model, t):
    x0 = jnp.zeros_like(t)
    xL = jnp.full_like(t, L)
    u_left = model(jnp.stack([x0, t], axis=-1))[:, 0]
    u_right = model(jnp.stack([xL, t], axis=-1))[:, 0]
    return jnp.mean(u_left**2) + jnp.mean(u_right**2)


def ic_fn(model, x):
    t0 = jnp.zeros_like(x)
    u, u_x, u_t, u_xx, u_tt = partials(model, x, t0)
    return jnp.mean((u - jnp.sin(k * x)) ** 2) + jnp.mean(u_t**2)


T = 2 * L / c  # one full period of cos(c k t)

# Interior collocation: a grid of (x, t) pairs, flattened to matching 1-D arrays.
xg = jnp.linspace(0, L, 60)
tg = jnp.linspace(0, T, 60)
X, T_grid = jnp.meshgrid(xg, tg, indexing="ij")
collocation = (X.ravel(), T_grid.ravel())

x_ic = jnp.linspace(0, L, 100)  # bottom edge t = 0
t_bc = jnp.linspace(0, T, 100)  # side edges x = 0, L

# ---------- MLP ----------
MLP_model = MLP([2, 32, 32, 32, 1], act_fun=nnx.silu, rngs=nnx.Rngs(0))
MLP_loss_history = train(
    MLP_model,
    loss=lambda model: loss_fn(
        model,
        collocation,
        residual=residual,
        ic_fn=lambda m: ic_fn(m, x_ic),
        bc_fn=lambda m: bc_fn(m, t_bc),
    ),
    steps=10000,
    lr=0.001,
)

# ---------- KANN ----------
KANN_model = KANN([2, 32, 32, 32, 1], rngs=nnx.Rngs(0))
KANN_loss_history = train(
    KANN_model,
    loss=lambda model: loss_fn(
        model,
        collocation,
        residual=residual,
        ic_fn=lambda m: ic_fn(m, x_ic),
        bc_fn=lambda m: bc_fn(m, t_bc),
    ),
    steps=10000,
    lr=0.001,
)

# Evaluate both models on the (x, t) grid.
XT = jnp.stack([X.ravel(), T_grid.ravel()], axis=-1)
U_MLP = MLP_model(XT)[:, 0].reshape(X.shape)
U_KANN = KANN_model(XT)[:, 0].reshape(X.shape)
U_exact = exact_solution(X, T_grid)


# ---------- Plotting ----------
FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

loss_dict = {"MLP_loss": MLP_loss_history, "KANN_loss": KANN_loss_history}
plot_losses(
    loss_dict,
    str(FIGS / "1D_wave_MLP_vs_KANN_loss.pdf"),
    "1D Wave Equation - MLP VS. KANN",
)

# Snapshots u(x, .) at a few fixed times.
for j in [0, len(tg) // 4, len(tg) // 2]:
    plot_solutions(
        xg,
        {"exact": U_exact[:, j], "MLP": U_MLP[:, j], "KANN": U_KANN[:, j]},
        "x",
        "u(x,t)",
        f"1D Wave at t = {float(tg[j]):.2f}",
        str(FIGS / f"1D_wave_snapshot_t{j}.pdf"),
    )

# Full space-time fields and their error against the exact solution.
for name, U in [("exact", U_exact), ("MLP", U_MLP), ("KANN", U_KANN)]:
    plot_field(
        X,
        T_grid,
        U,
        "x",
        "t",
        f"1D Wave - {name}",
        str(FIGS / f"1D_wave_field_{name}.pdf"),
        cbar_label="u(x,t)",
    )

for name, U in [("MLP", U_MLP), ("KANN", U_KANN)]:
    plot_field(
        X,
        T_grid,
        jnp.abs(U - U_exact),
        "x",
        "t",
        f"1D Wave - |{name} - exact|",
        str(FIGS / f"1D_wave_error_{name}.pdf"),
        cmap="magma",
        cbar_label="abs. error",
    )
