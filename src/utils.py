import jax
import jax.numpy as jnp


def derivatives(model, x, order=2):
    """Derivatives of a scalar field at the points x, shape (N, d).

    Returns (u, grad, hess, ...) up to `order`, with shapes (N,), (N, d),
    (N, d, d), ...

    The first derivative is taken in reverse mode and everything above it in
    forward mode. Reverse mode gets the gradient of a scalar output in a single
    pass, so the Hessian costs ~d passes; nesting jacfwd instead costs ~d**2,
    since the outer pass re-differentiates the inner one's d passes. For a KANN
    at d=3 -- where each pass drags the whole spline-basis recursion along --
    that is a ~10x difference per training step.
    """

    def u_scalar(xi):
        return model(xi[None, :])[0, 0]

    fns = [u_scalar]
    if order >= 1:
        fns.append(jax.grad(u_scalar))
        for _ in range(order - 1):
            fns.append(jax.jacfwd(fns[-1]))

    def all_derivs(xi):
        return tuple(f(xi) for f in fns)

    return jax.vmap(all_derivs)(x)


def laplacian(H, dims=None):
    dims = range(H.shape[-1]) if dims is None else dims
    return sum(H[:, i, i] for i in dims)
