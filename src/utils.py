import jax
import jax.numpy as jnp


def derivatives(model, x, order=2):
    def u_scalar(xi):
        return model(jnp.reshape(xi, (1, 1)))[0, 0]

    fns = [u_scalar]
    for _ in range(order):
        fns.append(jax.jacfwd(fns[-1]))  # jacrev (reverse jacobian), jacfwd

    def all_derivs(xi):
        return tuple(f(xi) for f in fns)

    return jax.vmap(all_derivs)(x)


def partials(model, x, t):
    def u_scalar(xi, ti):
        return model(jnp.stack([xi, ti]).reshape(1, 2))[0, 0]

    du_x = jax.grad(u_scalar, argnums=0)
    du_t = jax.grad(u_scalar, argnums=1)
    du_xx = jax.grad(du_x, argnums=0)
    du_tt = jax.grad(du_t, argnums=1)
    v = jax.vmap
    return (
        v(u_scalar)(x, t),
        v(du_x)(x, t),
        v(du_t)(x, t),
        v(du_xx)(x, t),
        v(du_tt)(x, t),
    )
