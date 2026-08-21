import jax.numpy as jnp


def loss_fn(model, t_collocation, residual, ic_fn):
    pde_loss = jnp.mean(residual(model, t_collocation) ** 2)
    return pde_loss + ic_fn(model)
