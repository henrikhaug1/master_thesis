import flax.nnx as nnx
import jax.numpy as jnp
import time
import optax

from dataclasses import dataclass


@dataclass
class TrainResult:
    loss_history: jnp.ndarray
    train_time: float
    best_loss: float
    n_steps: int
    stop_reason: str
    metric_history: jnp.ndarray | None = None
    metric_steps: jnp.ndarray | None = None


def train(
    model,
    loss,
    steps=5000,
    lr=1e-3,
    restore_best=True,
    check_every=100,
    rel_tol=1e-6,
    patience=5,
    verbose_every=500,
    metric_fn=None,
    tx=None,
    name="adam",
):
    tx = tx if tx is not None else optax.adam(lr)
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

    loss_history = []
    metric_history, metric_steps = [], []
    best_loss, best_params = float("inf"), None

    start = time.perf_counter()

    stagnant = 0
    stop_reason = "max_steps"
    ref = float("inf")

    for i in range(steps):
        checking = i % check_every == 0
        params = (
            nnx.to_pure_dict(nnx.state(model, nnx.Param))
            if restore_best and checking
            else None
        )
        if checking and metric_fn is not None:
            metric_steps.append(i)
            metric_history.append(float(metric_fn(model)))
        loss_val = train_step(model, optimizer)
        loss_history.append(loss_val)

        if not checking:
            continue

        if not jnp.isfinite(loss_val):
            stop_reason = "diverged"
            break

        l = float(loss_val)

        if l < best_loss:
            best_loss = l
            if restore_best:
                best_params = params

        stagnant = 0 if best_loss < ref * (1 - rel_tol) else stagnant + 1
        ref = best_loss
        if stagnant >= patience:
            stop_reason = "converged"
            break

        if verbose_every and i % verbose_every == 0:
            print(f"{name:<5} step {i:5d} | loss {l:.6e}")

    if best_params is not None:
        state = nnx.state(model, nnx.Param)
        nnx.replace_by_pure_dict(state, best_params)  # mutates in place
        nnx.update(model, state)

    train_time = time.perf_counter() - start

    return TrainResult(
        loss_history=jnp.asarray(loss_history),
        train_time=train_time,
        best_loss=best_loss,
        n_steps=len(loss_history),
        stop_reason=stop_reason,
        metric_history=jnp.asarray(metric_history) if metric_fn else None,
        metric_steps=jnp.asarray(metric_steps) if metric_fn else None,
    )


def train_adam_lbfgs(model, loss, adam_steps=6000, lbfgs_steps=4000, lr=1e-3, **kw):
    adam = train(model, loss, steps=adam_steps, tx=optax.adam(lr), name="adam", **kw)
    lbfgs = train(
        model, loss, steps=lbfgs_steps, tx=optax.lbfgs(), name="lbfgs", **kw
    )

    offset = adam.n_steps
    has_metric = adam.metric_history is not None
    return TrainResult(
        loss_history=jnp.concatenate([adam.loss_history, lbfgs.loss_history]),
        train_time=adam.train_time + lbfgs.train_time,
        best_loss=min(adam.best_loss, lbfgs.best_loss),
        n_steps=adam.n_steps + lbfgs.n_steps,
        stop_reason=f"adam: {adam.stop_reason}, lbfgs: {lbfgs.stop_reason}",
        metric_history=(
            jnp.concatenate([adam.metric_history, lbfgs.metric_history])
            if has_metric
            else None
        ),
        metric_steps=(
            jnp.concatenate([adam.metric_steps, lbfgs.metric_steps + offset])
            if has_metric
            else None
        ),
    )
