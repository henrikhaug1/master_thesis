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
):
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer):
        loss_val, grads = nnx.value_and_grad(loss)(model)
        optimizer.update(model, grads)
        return loss_val

    loss_history = []
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
            print(f"step {i:5d} | loss {l:.6e}")

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
    )
