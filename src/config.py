"""Every setting of an experiment, as plain data.

Experiments describe their models and training with these dataclasses instead
of code, so changing a setting is a one-line edit, and each result can store
the exact settings that produced it (see src.experiment).
"""

from dataclasses import dataclass, field

import flax.nnx as nnx
import jax.numpy as jnp

from src.bases import BSplineBasis, ChebyshevBasis
from src.pinns import KANN, MLP

ACTIVATIONS = {
    "silu": nnx.silu,
    "tanh": nnx.tanh,
    "gelu": nnx.gelu,
    "relu": nnx.relu,
    "sin": jnp.sin,
}

INITIALIZERS = {
    "lecun_normal": nnx.initializers.lecun_normal(),
    "glorot_normal": nnx.initializers.glorot_normal(),
    "glorot_uniform": nnx.initializers.glorot_uniform(),
    "he_normal": nnx.initializers.he_normal(),
    "he_uniform": nnx.initializers.he_uniform(),
}

BASES = {
    "bspline": BSplineBasis,
    "cheb": ChebyshevBasis,
}


# ---------- Training ----------


@dataclass(frozen=True)
class Stage:
    """One optimizer phase, e.g. Stage("adam", 5000, lr=1e-3).

    optimizer: a key of src.train.OPTIMIZERS ("adam", "adamw", "sgd", "soap",
        "lbfgs").
    steps: maximum number of steps; early stopping can end the stage sooner.
    lr: learning rate. None keeps the optimizer's own default -- a line search
        for L-BFGS, 3e-3 for SOAP. Adam, AdamW and SGD need one.
    options: further keyword arguments of the optax constructor, e.g.
        {"precondition_frequency": 5} for SOAP.
    """

    optimizer: str
    steps: int
    lr: float | None = None
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    """How every model in a comparison is trained.

    stages: optimizer phases, run in order; each starts from the parameters the
        previous one ended with.
    seeds: one training run per seed; results are reported over all of them.
    check_every: steps between checks of the loss and the error metric.
    rel_tol, patience: a stage stops early once the best loss has improved by
        less than a relative `rel_tol` for `patience` checks in a row.
    restore_best: end each stage on the parameters with the lowest checked loss.
    verbose_every: steps between progress prints; 0 silences them.
    """

    stages: tuple[Stage, ...]
    seeds: tuple[int, ...] = (0, 1, 2)
    check_every: int = 100
    rel_tol: float = 1e-6
    patience: int = 5
    restore_best: bool = True
    verbose_every: int = 500

    @property
    def name(self) -> str:
        """Default run-folder name, e.g. "soap6000-lbfgs4000"."""
        return "-".join(f"{s.optimizer}{s.steps}" for s in self.stages)


# ---------- Models ----------


@dataclass(frozen=True)
class MLPConfig:
    """Fully connected network with `activation` between the linear layers.

    widths: layer sizes including input and output, e.g. (1, 48, 48, 1).
    activation: a key of ACTIVATIONS.
    init: a key of INITIALIZERS, for the weights; biases start at zero.
        "lecun_normal" is the Flax default.
    """

    widths: tuple[int, ...]
    activation: str = "silu"
    init: str = "lecun_normal"

    def build(self, rngs: nnx.Rngs) -> MLP:
        return MLP(
            list(self.widths),
            act_fun=ACTIVATIONS[self.activation],
            kernel_init=INITIALIZERS[self.init],
            rngs=rngs,
        )


@dataclass(frozen=True)
class KANNConfig:
    """Kolmogorov-Arnold network: every edge is a learned sum of basis functions.

    widths: layer sizes including input and output, e.g. (1, 16, 16, 1).
    basis: a key of BASES.
    basis_options: keyword arguments of the basis class, used in every layer.
    input_options: overrides of `basis_options` for the first layer only --
        typically the domain of the PDE, which is known there but not in the
        hidden layers (see ChebyshevBasis).
    coeff_std: standard deviation of the initial basis coefficients.
    base_init: a key of INITIALIZERS, for the SiLU base weights.
    decay_in_forward: apply the basis' coefficient decay in the forward pass
        instead of to the initial coefficients.
    """

    widths: tuple[int, ...]
    basis: str = "bspline"
    basis_options: dict = field(default_factory=dict)
    input_options: dict = field(default_factory=dict)
    coeff_std: float = 0.1
    base_init: str = "he_uniform"
    decay_in_forward: bool = False

    def build(self, rngs: nnx.Rngs) -> KANN:
        basis = BASES[self.basis]
        return KANN(
            list(self.widths),
            basis_fn=lambda: basis(**self.basis_options),
            input_basis_fn=lambda: basis(
                **{**self.basis_options, **self.input_options}
            ),
            coeff_std=self.coeff_std,
            base_init=INITIALIZERS[self.base_init],
            decay_in_forward=self.decay_in_forward,
            rngs=rngs,
        )


ModelConfig = MLPConfig | KANNConfig
