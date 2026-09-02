import flax.nnx as nnx
import optax


def train(model, loss, steps=5000, lr=1e-3, restore_best=True):
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer):
        loss_val, grads = nnx.value_and_grad(loss)(model)
        optimizer.update(model, grads)
        return loss_val

    history = []
    best_loss, best_params = float("inf"), None
    for i in range(steps):
        params = nnx.to_pure_dict(nnx.state(model, nnx.Param)) if restore_best else None
        loss_val = train_step(model, optimizer)
        history.append(loss_val)
        if restore_best and float(loss_val) < best_loss:
            best_loss, best_params = float(loss_val), params
        if i % 500 == 0:
            print(f"step {i:5d} | loss {float(loss_val):.6e}")

    if best_params is not None:
        state = nnx.state(model, nnx.Param)
        nnx.replace_by_pure_dict(state, best_params)
        nnx.update(model, state)
        print(f"restored best params | loss: {best_loss:.6e}")

    return history
