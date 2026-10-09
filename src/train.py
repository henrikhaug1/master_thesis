"""Training: a run is a sequence of optimizer stages (see src.config.Stage)."""

import time
from dataclasses import dataclass

import flax.nnx as nnx
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
    # lets every other optimizer ignore them.
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

    Returns (best checked loss, stop reason).
    """
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def value_fn(p):
            return loss(nnx.merge(graphdef, p, rest))

        loss_val, grads = nnx.value_and_grad(loss)(model)
        optimizer.update(
            model,
            grads,
            value=loss_val,
            grad=nnx.as_pure(nnx.state(grads, nnx.Param)),
            value_fn=value_fn,
        )
        return loss_val

    offset = len(log["loss"])
    best_loss, best_params = float("inf"), None
    ref, stagnant, stop_reason = float("inf"), 0, "max_steps"

    for i in range(stage.steps):
        checking = i % cfg.check_every == 0
        if checking:
            # Snapshot before the step: the loss it returns belongs to these.
            params = (
                nnx.to_pure_dict(nnx.state(model, nnx.Param))
                if cfg.restore_best
                else None
            )
            if metric_fn is not None:
                log["metric_steps"].append(offset + i)
                log["metric"].append(float(metric_fn(model)))

        loss_val = train_step(model, optimizer)
        log["loss"].append(loss_val)
        if not checking:
            continue

        if not jnp.isfinite(loss_val):
            stop_reason = "diverged"
            break

        l = float(loss_val)
        if l < best_loss:
            best_loss, best_params = l, params

        stagnant = 0 if best_loss < ref * (1 - cfg.rel_tol) else stagnant + 1
        ref = best_loss
        if stagnant >= cfg.patience:
            stop_reason = "converged"
            break

        if cfg.verbose_every and i % cfg.verbose_every == 0:
            print(f"{stage.optimizer:<5} step {i:5d} | loss {l:.6e}")

    if best_params is not None:
        state = nnx.state(model, nnx.Param)
        nnx.replace_by_pure_dict(state, best_params)
        nnx.update(model, state)

    return best_loss, stop_reason
