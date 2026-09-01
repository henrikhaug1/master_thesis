import jax.numpy as jnp


def _to_reference(x, domain, scale=1.0):
    """
    Function that maps any domain to [-1, 1].

    With `domain=(low, high)` the map is affine, and exact for inputs that stay
    inside the domain. With `domain=None` the range of x is unknown, so it is
    squashed with tanh(x / scale) instead -- `scale` has to be wide enough that
    x does not land in the flat tails, where every basis function collapses onto
    the same value and its derivative underflows to zero.
    """
    if domain is None:
        return jnp.tanh(x / scale)
    else:
        low, high = domain
        return 2.0 * (x - low) / (high - low) - 1.0


class BSplineBasis:
    def __init__(self, grid_size=5, spline_order=3, grid_range=(-0.5, 10.5)):
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.grid = (
            jnp.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        )
        self.n_basis = grid_size + spline_order

    def __call__(self, x):
        grid = self.grid
        x = x[..., None]
        bases = ((x >= grid[:-1]) & (x < grid[1:])).astype(x.dtype)
        for p in range(1, self.spline_order + 1):
            left = (
                (x - grid[: -(p + 1)])
                / (grid[p:-1] - grid[: -(p + 1)] + 1e-12)
                * bases[..., :-1]
            )
            right = (
                (grid[p + 1 :] - x)
                / (grid[p + 1 :] - grid[1:-p] + 1e-12)
                * bases[..., 1:]
            )
            bases = left + right
        return bases


class ChebyshevBasis:
    """Chebyshev polynomials T_0..T_degree of the first kind.

    Args:
        degree: highest polynomial degree, so n_basis = degree + 1.
        domain: (low, high) range the inputs are affinely mapped from. Pass the
            real range whenever it is known -- i.e. for the input layer, where
            it is the domain of the PDE. Passing None instead squashes the
            inputs with tanh, which saturates: on t in [0, 10] every T_n is
            equal to 1 within 1e-3 past t = 4, and dT_n/dt underflows to zero,
            so the basis contributes nothing to the residual.
        scale: width of the tanh squash used when domain is None. Only relevant
            for hidden layers, whose range is not known ahead of time and where
            an affine map is not an option either: T_n diverges like z**n once
            |z| > 1.
    """

    def __init__(self, degree=5, domain=None, scale=1.0):
        self.degree = degree
        self.domain = domain
        self.scale = scale
        self.n_basis = degree + 1

    def __call__(self, x):
        z = _to_reference(x, domain=self.domain, scale=self.scale)[..., None]
        polynomials = [jnp.ones_like(z), z]

        for n in range(2, self.n_basis):
            polynomials.append(2 * z * polynomials[n - 1] - polynomials[n - 2])
        return jnp.concatenate(polynomials[: self.n_basis], axis=-1)
