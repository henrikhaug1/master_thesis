"""Every setting of an experiment, as plain data.

Experiments describe their models and training with these dataclasses instead
of code, so changing a setting is a one-line edit, and each result can store
the exact settings that produced it (see src.experiment).
"""

from dataclasses import dataclass, field

import flax.nnx as nnx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.bases import BSplineBasis, ChebyshevBasis  # noqa: E402 -- after the x64 switch
from src.pinns import KANN, MLP  # noqa: E402
from src.sampling import (  # noqa: E402
    Domain,
    Resampler,
    graded_points,
    grid_points,
    random_points,
    residual_density_points,
)

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
    check_every: steps between records of the error metric and convergence
        checks. The best loss and divergence are tracked at every step.
    rel_tol, patience: a stage stops early once the best loss has improved by
        less than a relative `rel_tol` for `patience` checks in a row.
    restore_best: end each stage on the parameters with the lowest loss of any
        step, rather than on the last ones.
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


# ---------- Collocation ----------
# Each strategy places the interior points for a src.sampling.Domain:
# points(domain) returns them with their loss weights (None weighs them all
# equally), and tag names the strategy in run folders. RADCollocation also
# moves its points during training, through resampler().


@dataclass(frozen=True)
class GridCollocation:
    """An evenly spaced grid of n points per axis over the domain's box, keeping
    those inside the domain. The baseline, so its tag is empty: run folders
    from before the strategies existed all used it."""

    n: int

    @property
    def tag(self) -> str:
        return ""

    def points(self, domain: Domain):
        return grid_points(domain.lo, domain.hi, self.n, domain.inside), None


@dataclass(frozen=True)
class RandomCollocation:
    """n points drawn uniformly at random inside the domain, from `seed`.

    The seed is separate from the training seeds, so every model and seed
    trains on the same points.
    """

    n: int
    seed: int = 0

    @property
    def tag(self) -> str:
        return f"random-s{self.seed}"

    def points(self, domain: Domain):
        key = jax.random.key(self.seed)
        return random_points(domain.lo, domain.hi, self.n, key, domain.inside), None


@dataclass(frozen=True)
class GradedCollocation:
    """n_r x n_theta points graded toward domain.center (see graded_points),
    weighted by the area each stands for.

    beta: grading exponent; 1 is an ordinary polar grid, larger packs the center.
    s_min: the innermost fraction of the way out, to keep points off the center.
    weighted: weigh each point by its area, so the loss stays an integral over
        the domain and the grading only sharpens the quadrature. False weighs
        all points equally, so the loss leans on the center where they gather.
    """

    n_r: int
    n_theta: int
    beta: float = 2.0
    s_min: float = 0.0
    weighted: bool = True

    @property
    def tag(self) -> str:
        return f"graded-b{self.beta:g}" + ("" if self.weighted else "-unweighted")

    def points(self, domain: Domain):
        if domain.center is None:
            raise ValueError("GradedCollocation needs a Domain with a center")
        pts, area = graded_points(
            domain.center,
            *domain.theta_range,
            domain.r_max_fn,
            self.n_r,
            self.n_theta,
            self.beta,
            self.s_min,
        )
        return pts, area if self.weighted else None


@dataclass(frozen=True)
class RADCollocation:
    """n points, redrawn every `every` steps where the residual is large: the
    residual-based adaptive distribution (RAD) of Wu et al., CMAME 403 (2023).

    Each draw takes n of n_candidates fresh uniform random points, with
    probability proportional to |r|^k / mean(|r|^k) + c. k sharpens the
    preference for large residuals; c keeps a uniform share, so a large c is
    plain random resampling. Training starts on n uniform random points, and
    all points weigh the same, so the loss leans on where they gather.

    seed: of the starting points and of every draw, shared by all models and
        training seeds.
    """

    n: int
    n_candidates: int = 30_000
    k: float = 1.0
    c: float = 1.0
    every: int = 500
    seed: int = 0

    @property
    def tag(self) -> str:
        return f"rad-k{self.k:g}-c{self.c:g}"

    def points(self, domain: Domain):
        return RandomCollocation(self.n, self.seed).points(domain)

    def resampler(self, domain: Domain, residual) -> Resampler:
        """The draws for src.train.train, with `residual(model, pts)` the
        same residual the loss uses."""
        residual = nnx.jit(residual)

        def draw(model, key):
            cand_key, pick_key = jax.random.split(key)
            candidates = random_points(
                domain.lo, domain.hi, self.n_candidates, cand_key, domain.inside
            )
            return residual_density_points(
                model, residual, candidates, self.n, pick_key, self.k, self.c
            )

        pts, _ = self.points(domain)
        return Resampler(pts, draw, self.every, self.seed)


CollocationConfig = (
    GridCollocation | RandomCollocation | GradedCollocation | RADCollocation
)


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
        hidden layers (see ChebyshevBasis and BSplineBasis).
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
