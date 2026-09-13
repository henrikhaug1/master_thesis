import jax
import jax.numpy as jnp


def derivatives(model, x, order=2):
    def u_scalar(xi):
        return model(xi[None, :])[0, 0]

    fns = [u_scalar]
    for _ in range(order):
        fns.append(jax.jacfwd(fns[-1]))

    def all_derivs(xi):
        return tuple(f(xi) for f in fns)

    return jax.vmap(all_derivs)(x)


def laplacian(H, dims=None):
    dims = range(H.shape[-1]) if dims is None else dims
    return sum(H[:, i, i] for i in dims)
