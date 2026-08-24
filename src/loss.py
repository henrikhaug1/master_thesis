import jax.numpy as jnp


def loss_fn(model, collocation, residual, ic_fn=None, bc_fn=None):
    if ic_fn is None and bc_fn is None:
        raise ValueError("Define boundary- or initial conditions")
    coords = collocation if isinstance(collocation, tuple) else (collocation,)
    loss = jnp.mean(residual(model, *coords) ** 2)
    if ic_fn is not None:
        loss += ic_fn(model)
    if bc_fn is not None:
        loss += bc_fn(model)
    return loss
