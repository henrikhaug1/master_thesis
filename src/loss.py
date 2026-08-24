import jax.numpy as jnp


def loss_fn(model, t_collocation, residual, ic_fn=None, bc_fn=None):
    pde_loss = jnp.mean(residual(model, t_collocation) ** 2)
    loss = pde_loss
    if ic_fn is None and bc_fn is None:
        raise ValueError("Define boundary- or initial conditions")
    if ic_fn is not None:
        loss += ic_fn(model)
    if bc_fn is not None:
        loss += bc_fn(model)
    return loss
