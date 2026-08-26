import jax.numpy as jnp


def _to_reference(x, domain):
    """
    Function that maps any domain to [-1, 1]
    """
    if domain is None:
        return jnp.tanh(x)
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
