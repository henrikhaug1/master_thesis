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
        res = train(model, loss, **train_kw)
        hist = np.asarray(res.loss_history, dtype=float)
        u = predict_fn(model)
        out.append(
            {
                "seed": s,
                "final_loss": float(hist[-1]),
                # `train` restores the parameters that reached `best_loss`, so
                # this is the loss the metrics below belong to -- not
                # hist.min(), which may dip at a step never checkpointed.
                "best_loss": float(res.best_loss),
                "rel_l2": rel_l2_error(u, u_exact),
                "max_err": max_error(u, u_exact),
                "n_params": n_params(model),
                "train_time": float(res.train_time),
                "n_steps": int(res.n_steps),
                "stop_reason": res.stop_reason,
                "history": hist,
                "u": np.asarray(u),
            }
        )
    return out


def stack_histories(runs: Sequence[Run]) -> np.ndarray:
    """Loss histories as one (n_runs, max_steps) array, NaN-padded.

    Early stopping means two seeds of the same model can stop at different
    steps, so the histories are ragged and cannot be stacked directly. Padding
    with NaN keeps the step axis aligned; plotting and aggregation use the
    nan-aware reductions.
    """
    hists = [np.asarray(r["history"], dtype=float) for r in runs]
    out = np.full((len(hists), max(len(h) for h in hists)), np.nan)
    for i, h in enumerate(hists):
        out[i, : len(h)] = h
    return out


def summarize(results: Results) -> None:
    header = (
        f"{'model':<18}{'params':>8}{'rel L2 (mean+-std)':>26}"
        f"{'best loss':>14}{'steps':>8}{'time [s]':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, runs in results.items():
        rel = np.array([r["rel_l2"] for r in runs])
        best = np.array([r["best_loss"] for r in runs])
        steps = np.array([r["n_steps"] for r in runs])
        times = np.array([r["train_time"] for r in runs])
        print(
            f"{name:<18}{runs[0]['n_params']:>8}"
            f"{rel.mean():>14.3e} +-{rel.std():>9.1e}"
            f"{np.median(best):>14.3e}"
            f"{int(np.median(steps)):>8}{np.median(times):>10.1f}"
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
            "histories": stack_histories(runs),
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
        {n: stack_histories(runs) for n, runs in results.items()},
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
