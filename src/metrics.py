import jax.numpy as jnp


def l2_error(u_pred, u_exact):
    return float(jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)))


def rel_l2_error(u_pred, u_exact):
    return float(jnp.linalg.norm(u_pred - u_exact) / jnp.linalg.norm(u_exact))


def max_error(u_pred, u_exact):
    return jnp.max(jnp.abs(u_pred - u_exact))
