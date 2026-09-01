from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from jax import Array

from src.train import train
from src.metrics import rel_l2_error, max_error, n_params
from src.plotting import plot_loss_bands, plot_solutions

ModelFn = Callable[[nnx.Rngs], nnx.Module]
LossFn = Callable[[nnx.Module], Array]
PredictFn = Callable[[nnx.Module], Array]
Run = dict[str, Any]
Results = dict[str, list[Run]]


def run_seeds(
    model_fn: ModelFn,
    loss: LossFn,
    predict_fn: PredictFn,
    u_exact: Array,
    seeds: Sequence[int] = (0, 1, 2),
    **train_kw: Any,
) -> list[Run]:
    out: list[Run] = []
    for s in seeds:
        print(f"\n ----- Seed: {s} -----")
        model = model_fn(nnx.Rngs(s))
        hist = train(model, loss, **train_kw)
        u = predict_fn(model)
        out.append(
            {
                "seed": s,
                "final_loss": float(hist[-1]),
                "best_loss": float(min(hist)),
                "rel_l2": rel_l2_error(u, u_exact),
                "max_err": max_error(u, u_exact),
                "n_params": n_params(model),
                "history": np.asarray([float(h) for h in hist]),
                "u": np.asarray(u),
            }
        )
    return out


def summarize(results: Results) -> None:
    header = f"{'model':<18}{'params':>8}{'rel L2 (mean+-std)':>26}{'best loss':>14}"
    print(header)
    print("-" * len(header))
    for name, runs in results.items():
        rel = np.array([r["rel_l2"] for r in runs])
        best = np.array([r["best_loss"] for r in runs])
        print(
            f"{name:<18}{runs[0]['n_params']:>8}"
            f"{rel.mean():>14.3e} +-{rel.std():>9.1e}"
            f"{np.median(best):>14.3e}"
        )


def compare_models(
    models: Mapping[str, ModelFn],
    loss: LossFn,
    predict_fn: PredictFn,
    u_exact: Array,
    x: Array | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    out_dir: str | Path = ".",
    title: str = "",
    x_label: str = "x",
    y_label: str = "u(x)",
    **train_kw: Any,
) -> Results:
    """Run every model over `seeds`, save raw results, and plot the comparison.

    Args:
        models: {name: model_fn}, where model_fn(rngs) -> model.
        loss: model -> scalar, the same closure `train` expects.
        predict_fn: model -> predictions aligned with `u_exact`.
        u_exact: reference solution used for the error metrics.
        x: 1-D grid the predictions are plotted against. Pass None for problems
            with more than one input (the metrics still run on the flattened
            predictions; plot the fields yourself from the returned runs).
        out_dir: results/*.npz and figs/*.pdf are written under here.

    Returns {name: [run dicts]}.
    """
    out_dir = Path(out_dir)
    figs, results_dir = out_dir / "figs", out_dir / "results"
    figs.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    results: Results = {}
    for name, model_fn in models.items():
        print(f"\n ----- MODEL: {name} -----")
        runs = run_seeds(model_fn, loss, predict_fn, u_exact, seeds=seeds, **train_kw)
        results[name] = runs
        payload = {
            "histories": np.stack([r["history"] for r in runs]),
            "predictions": np.stack([r["u"] for r in runs]),
            "seeds": np.array([r["seed"] for r in runs]),
            "rel_l2": np.array([r["rel_l2"] for r in runs]),
            "n_params": runs[0]["n_params"],
            "u_exact": np.asarray(u_exact),
        }
        if x is not None:
            payload["x"] = np.asarray(x)
        np.savez(results_dir / f"{name}.npz", **payload)

    slug = title.replace(" ", "_") or "comparison"

    plot_loss_bands(
        {n: np.stack([r["history"] for r in runs]) for n, runs in results.items()},
        str(figs / f"{slug}_loss.pdf"),
        f"{title} - loss (median over {len(seeds)} seeds)",
    )

    if x is not None:
        curves = {"exact": jnp.asarray(u_exact)}
        for name, runs in results.items():
            curves[name] = np.median(np.stack([r["u"] for r in runs]), axis=0)
        plot_solutions(
            x, curves, x_label, y_label, title, str(figs / f"{slug}_solutions.pdf")
        )

    print()
    summarize(results)
    return results
