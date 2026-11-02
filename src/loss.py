import jax.numpy as jnp


def dirichlet_loss(model, pts, values=0.0):
    """Mean squared misfit between u and the prescribed `values` at `pts`."""
    u = model(pts)[:, 0]
    return jnp.mean((u - values) ** 2)


def loss_fn(
    model,
    collocation,
    residual,
    ic_fn=None,
    bc_fn=None,
    hard_constraints=False,
    w_res=1.0,
    w_ic=1.0,
    w_bc=1.0,
    weights=None,
):
    if ic_fn is None and bc_fn is None and not hard_constraints:
        raise ValueError(
            "Define boundary- or initial conditions, or pass hard_constraints=True "
        )
    r2 = residual(model, collocation) ** 2
    if weights is None:
        loss = w_res * jnp.mean(r2)
    else:
        loss = w_res * jnp.sum(weights * r2) / jnp.sum(weights)
    if ic_fn is not None:
        loss += w_ic * ic_fn(model)
    if bc_fn is not None:
        loss += w_bc * bc_fn(model)
    return loss
