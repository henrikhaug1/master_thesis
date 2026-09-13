import jax.numpy as jnp


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
):
    if ic_fn is None and bc_fn is None and not hard_constraints:
        raise ValueError(
            "Define boundary- or initial conditions, or pass hard_constraints=True "
        )
    loss = w_res * jnp.mean(residual(model, collocation) ** 2)
    if ic_fn is not None:
        loss += w_ic * ic_fn(model)
    if bc_fn is not None:
        loss += w_bc * bc_fn(model)
    return loss
