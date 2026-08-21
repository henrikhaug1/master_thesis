# src/utils.py
import jax, jax.numpy as jnp


def derivatives(model, t, order=2):
    def u_scalar(ti):
        return model(jnp.reshape(ti, (1, 1)))[0, 0]

    fns = [u_scalar]
    for _ in range(order):
        fns.append(jax.grad(fns[-1]))
    return [jax.vmap(f)(t) for f in fns]
