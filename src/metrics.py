import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax import Array


def l2_error(u_pred: Array, u_exact: Array) -> float:
    return float(jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)))


def rel_l2_error(u_pred: Array, u_exact: Array) -> float:
    return float(jnp.linalg.norm(u_pred - u_exact) / jnp.linalg.norm(u_exact))


def max_error(u_pred: Array, u_exact: Array) -> float:
    return float(jnp.max(jnp.abs(u_pred - u_exact)))


def n_params(model: nnx.Module) -> int:
    return sum(
        a.size for a in jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    )
