import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_solutions(
    x, dict: dict, x_label: str, y_label: str, title: str, filename: str
):
    x = jnp.asarray(x)
    plt.figure(figsize=(8, 4))
    for label, y in dict.items():
        plt.plot(x, jnp.asarray(y), label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_losses(histories: dict, filename: str, title: str = "Training loss"):
    plt.figure(figsize=(8, 4))
    for label, hist in histories.items():
        plt.plot(jnp.asarray(hist), label=label)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
