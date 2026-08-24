import flax.nnx as nnx
import optax


def train(model, loss, steps=5000, lr=1e-3):
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer):
        loss_val, grads = nnx.value_and_grad(loss)(model)
        optimizer.update(model, grads)
        return loss_val

    history = []
    for i in range(steps):
        loss_val = train_step(model, optimizer)
        history.append(loss_val)
        if i % 500 == 0:
            print(f"step {i:5d} | loss {float(loss_val):.6e}")
    return history
