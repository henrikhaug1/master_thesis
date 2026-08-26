import flax.nnx as nnx
import jax
import jax.numpy as jnp
from src.bases import BSplineBasis


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
        basis=None,
        *,
        rngs: nnx.Rngs,
    ):
        self.basis = basis if basis is not None else BSplineBasis()
        key1, key2 = jax.random.split(rngs.params())

        self.base_weight = nnx.Param(
            nnx.initializers.he_uniform()(key1, (in_features, out_features))
        )
        self.coeff = nnx.Param(
            0.1
            * jax.random.normal(key2, (in_features, out_features, self.basis.n_basis))
        )

    def __call__(self, x):
        base_out = jax.nn.silu(x) @ self.base_weight
        phi = self.basis(x)
        return base_out + jnp.einsum("bic, ioc->bo", phi, self.coeff)


class KANN(nnx.Module):
    def __init__(self, layer_sizes, basis_fn=BSplineBasis, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            KANLinear(layer_sizes[i], layer_sizes[i + 1], basis=basis_fn(), rngs=rngs)
            for i in range(len(layer_sizes) - 1)
        )

    def __call__(self, t):
        x = t
        for layer in self.layers:
            x = layer(x)
        return x
