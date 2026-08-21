import flax.nnx as nnx
import jax, jax.numpy as jnp
import optax


class MLP(nnx.Module):
    def __init__(self, layer_sizes: list, act_fun=nnx.tanh, *, rngs: nnx.Rngs):
        self.act_fun = act_fun
        self.layers = nnx.List(
            nnx.Linear(layer_sizes[i], layer_sizes[i + 1], rngs=rngs)
            for i in range(len(layer_sizes) - 1)
        )

    def __call__(self, t):
        x = t
        for layer in self.layers[:-1]:
            x = self.act_fun(layer(x))  # activation on hidden layers
        return self.layers[-1](x)  # last layer linear


class KANLinear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size=5,
        spline_order=3,
        grid_range=(-0.5, 10.5),
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            jnp.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        )
        self.grid = grid  # This grid is knot locations of the B-Spline

        key1, key2 = jax.random.split(rngs.params())

        self.base_weight = nnx.Param(
            nnx.initializers.he_uniform()(key1, (in_features, out_features))
        )
        self.spline_weight = nnx.Param(
            0.1
            * jax.random.normal(
                key2, (in_features, out_features, grid_size + spline_order)
            )
        )

    def b_splines(self, x):
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

    def __call__(self, x):
        base_out = jax.nn.silu(x) @ self.base_weight
        spline = self.b_splines(x)
        spline_out = jnp.einsum("bic,ioc->bo", spline, self.spline_weight)
        return base_out + spline_out


class KANN(nnx.Module):
    def __init__(self, layer_sizes, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            KANLinear(layer_sizes[i], layer_sizes[i + 1], rngs=rngs)
            for i in range(len(layer_sizes) - 1)
        )

    def __call__(self, t):
        x = t
        for layer in self.layers:
            x = layer(x)
        return x
