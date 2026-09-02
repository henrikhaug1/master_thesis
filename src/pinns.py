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
    def __init__(
        self,
        layer_sizes,
        basis_fn=BSplineBasis,
        input_basis_fn=None,
        *,
        rngs: nnx.Rngs,
    ):
        input_basis_fn = input_basis_fn or basis_fn
        self.layers = nnx.List(
            KANLinear(
                layer_sizes[i],
                layer_sizes[i + 1],
                basis=(input_basis_fn if i == 0 else basis_fn)(),
                rngs=rngs,
            )
            for i in range(len(layer_sizes) - 1)
        )

    def __call__(self, t):
        x = t
        for layer in self.layers:
            x = layer(x)
        return x


class HardConstraint(nnx.Module):
    """Wrap a model so constraints hold by construction: u = g(x) + phi(x) * net(x).

    `phi` vanishes exactly where the constraint is imposed -- phi(x) = 4x(1-x)
    on [0, 1] pins u(0) = u(1) = 0 -- and `g` supplies the value taken there,
    defaulting to zero. Constraints enforced this way need no penalty term, so
    they cannot be traded off against the residual the way a soft term can.

    Both are plain callables mapping the raw input (N, d) to something that
    broadcasts against (N, 1), and are static under jit like `MLP.act_fun`.
    Scale phi so max|phi| is about 1: a phi peaking well above 1 rescales the
    output and the residual with it.

    Args:
        net: the model being wrapped; its parameters are the only ones here.
        phi: (N, d) -> (N, 1), zero where the constraint applies.
        g: (N, d) -> (N, 1), the constrained value. None means zero.
    """

    def __init__(self, net, phi, g=None):
        self.net = net
        self.phi = phi
        self.g = g

    def __call__(self, x):
        u = self.phi(x) * self.net(x)
        return u if self.g is None else u + self.g(x)
