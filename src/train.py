"""Training: a run is a sequence of optimizer stages (see src.config.Stage)."""

import math
import time
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from soap_jax import soap

from src.config import Stage, TrainConfig

# Each entry is called with Stage.lr as `learning_rate` plus Stage.options.
OPTIMIZERS = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "soap": soap,
    "lbfgs": optax.lbfgs,
}


@dataclass
class TrainResult:
    loss_history: jnp.ndarray  # one entry per step, all stages back to back
    metric_history: jnp.ndarray  # metric_fn at metric_steps; empty without one
    metric_steps: jnp.ndarray
    best_loss: float
    train_time: float
    stage_ends: tuple[int, ...]  # cumulative step at which each stage stopped
    stop_reasons: tuple[str, ...]  # per stage: "max_steps", "converged" or "diverged"


def make_optimizer(stage: Stage) -> optax.GradientTransformationExtraArgs:
    if stage.optimizer not in OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {stage.optimizer!r}; choose from {list(OPTIMIZERS)}"
        )
    kwargs = dict(stage.options)
    if stage.lr is not None:
        kwargs["learning_rate"] = stage.lr
    # The train step passes value/grad/value_fn, which only L-BFGS uses; this
    # lets every other optimizer ignore them. It leaves the state unchanged,
    # so value_and_grad_from_state still finds the line search's values.
    return optax.with_extra_args_support(OPTIMIZERS[stage.optimizer](**kwargs))


def train(model, loss, cfg: TrainConfig, metric_fn=None) -> TrainResult:
    """Train `model` in place on `loss`, running `cfg.stages` in order.

    `metric_fn(model)`, typically the error against the exact solution, is
    recorded at every check without affecting training.
    """
    optimizers = [make_optimizer(s) for s in cfg.stages]  # catch typos up front
    log = {"loss": [], "metric": [], "metric_steps": []}
    best_losses, stage_ends, stop_reasons = [], [], []

    start = time.perf_counter()
    for stage, tx in zip(cfg.stages, optimizers, strict=True):
        best_loss, stop_reason = _train_stage(
            model, loss, stage, tx, cfg, metric_fn, log
        )
        best_losses.append(best_loss)
        stage_ends.append(len(log["loss"]))
        stop_reasons.append(stop_reason)

    return TrainResult(
        loss_history=jnp.asarray(log["loss"]),
        metric_history=jnp.asarray(log["metric"]),
        metric_steps=jnp.asarray(log["metric_steps"], dtype=int),
        best_loss=min(best_losses),
        train_time=time.perf_counter() - start,
        stage_ends=tuple(stage_ends),
        stop_reasons=tuple(stop_reasons),
    )


def _train_stage(model, loss, stage, tx, cfg, metric_fn, log):
    """Up to `stage.steps` steps of one optimizer, appending to `log`.

    The parameters are trained as a plain pytree and written back into `model`
    only where it is read: before metric_fn and at the end of the stage.

    Returns (lowest loss of any step, stop reason).
    """
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    # Raw arrays instead of Param wrappers, which SOAP's tree_map trips over.
    params = nnx.as_pure(params)

    def value_fn(p):
        return loss(nnx.merge(graphdef, p, rest))

    # L-BFGS's line search has already evaluated the loss and gradient at the
    # point it accepted, which is where the next step starts; reusing them
    # saves one evaluation per step. Other optimizers keep no such state.
    if stage.optimizer == "lbfgs":
        value_and_grad = optax.value_and_grad_from_state(value_fn)
    else:
        grad_fn = jax.value_and_grad(value_fn)

        def value_and_grad(p, state):
            return grad_fn(p)

    @jax.jit
    def train_step(params, opt_state):
        loss_val, grads = value_and_grad(params, state=opt_state)
        updates, opt_state = tx.update(
            grads, opt_state, params, value=loss_val, grad=grads, value_fn=value_fn
        )
        return optax.apply_updates(params, updates), opt_state, loss_val

    opt_state = tx.init(params)
    offset = len(log["loss"])
    best_loss, best_params = float("inf"), None
    ref, stagnant, stop_reason = float("inf"), 0, "max_steps"

    for i in range(stage.steps):
        checking = i % cfg.check_every == 0
        if checking and metric_fn is not None:
            nnx.update(model, params)
            log["metric_steps"].append(offset + i)
            log["metric"].append(float(metric_fn(model)))

        # The loss a step returns belongs to the parameters it started from.
        # The arrays are immutable, so keeping a reference is enough.
        before = params
        params, opt_state, loss_val = train_step(params, opt_state)
        loss_now = float(loss_val)
        log["loss"].append(loss_now)

        if not math.isfinite(loss_now):
            stop_reason = "diverged"
            break
        if loss_now < best_loss:
            best_loss = loss_now
            if cfg.restore_best:
                best_params = before
        if not checking:
            continue

        stagnant = 0 if best_loss < ref * (1 - cfg.rel_tol) else stagnant + 1
        ref = best_loss
        if stagnant >= cfg.patience:
            stop_reason = "converged"
            break

        if cfg.verbose_every and i % cfg.verbose_every == 0:
            print(f"{stage.optimizer:<5} step {i:5d} | loss {loss_now:.6e}")

    nnx.update(model, params if best_params is None else best_params)
    return best_loss, stop_reason
