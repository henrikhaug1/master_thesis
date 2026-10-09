import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
from jax import Array

from src.config import ModelConfig, TrainConfig
from src.metrics import max_error, n_params, rel_l2_error
from src.plotting import plot_loss_bands, plot_solution_grid
from src.train import train

Run = dict[str, Any]
Results = dict[str, list[Run]]

ARRAYS = ("loss_history", "rel_l2_history", "metric_steps", "u")


def compare_models(
    models: Mapping[str, ModelConfig],
    train_cfg: TrainConfig,
    loss: Callable[[nnx.Module], Array],
    predict_fn: Callable[[nnx.Module], Array],
    u_exact: Array,
    out_dir: str | Path,
    title: str,
    *,
    problem: Mapping[str, Any] | None = None,
    wrap: Callable[[nnx.Module], nnx.Module] | None = None,
    overrides: Mapping[str, TrainConfig] | None = None,
    run_name: str | None = None,
    x: Array | None = None,
    x_label: str = "x",
    y_label: str = "u(x)",
) -> tuple[Results, Path]:
    run_dir = Path(out_dir) / "runs" / (run_name or train_cfg.name)
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    results: Results = {}
    for name, model_cfg in models.items():
        cfg = (overrides or {}).get(name, train_cfg)
        settings = {
            "model": {"type": type(model_cfg).__name__, **asdict(model_cfg)},
            "train": asdict(cfg),
            "problem": dict(problem or {}),
            "wrap": None if wrap is None else wrap.__name__,
        }
        print(f"\n ----- MODEL: {name} -----")
        runs = run_model(model_cfg, cfg, loss, predict_fn, u_exact, wrap)
        _save(run_dir / "results" / name, settings, runs, u_exact, x)
        results[name] = runs

    slug = title.replace(" ", "_")
    plot_loss_bands(
        {n: _pad([r["loss_history"] for r in runs]) for n, runs in results.items()},
        str(figs / f"{slug}_loss.pdf"),
        f"{title} - loss (median over seeds)",
    )
    curves = {n: _common_steps(runs) for n, runs in results.items()}
    plot_loss_bands(
        {n: c for n, (_, c) in curves.items()},
        str(figs / f"{slug}_rel_l2.pdf"),
        f"{title} - relative L2 error (median over seeds)",
        steps={n: s for n, (s, _) in curves.items()},
        y_label="rel. L2 error",
    )
    if x is not None:
        plot_solution_grid(
            x,
            u_exact,
            {n: median_prediction(runs) for n, runs in results.items()},
            x_label,
            y_label,
            title,
            str(figs / f"{slug}_solutions.pdf"),
        )

    table = summarize(results)
    print("\n" + table)
    (run_dir / "summary.txt").write_text(table + "\n")
    return results, run_dir


def run_model(model_cfg, train_cfg, loss, predict_fn, u_exact, wrap=None) -> list[Run]:
    """Build and train one model per seed of `train_cfg`."""
    runs = []
    for seed in train_cfg.seeds:
        print(f"\n ----- seed {seed} -----")
        model = model_cfg.build(nnx.Rngs(seed))
        if wrap is not None:
            model = wrap(model)
        res = train(
            model,
            loss,
            train_cfg,
            metric_fn=lambda m: rel_l2_error(predict_fn(m), u_exact),
        )
        u = predict_fn(model)
        runs.append(
            {
                "seed": seed,
                "n_params": n_params(model),
                "rel_l2": rel_l2_error(u, u_exact),
                "max_err": max_error(u, u_exact),
                "best_loss": res.best_loss,
                "n_steps": len(res.loss_history),
                "train_time": res.train_time,
                "stage_ends": list(res.stage_ends),
                "stop_reasons": list(res.stop_reasons),
                "loss_history": res.loss_history,
                "rel_l2_history": res.metric_history,
                "metric_steps": res.metric_steps,
                "u": u,
            }
        )
    return runs


def median_prediction(runs: list[Run]) -> Array:
    return jnp.median(jnp.stack([r["u"] for r in runs]), axis=0)


def summarize(results: Results) -> str:
    """Table with one row per model, aggregated over its seeds."""
    header = (
        f"{'model':<26}{'params':>8}{'rel L2 (mean +- std)':>24}"
        f"{'best loss':>12}{'steps':>8}{'time [s]':>10}"
    )
    rows = [header, "-" * len(header)]
    for name, runs in results.items():
        rel, best, steps, time = (
            jnp.array([r[k] for r in runs])
            for k in ("rel_l2", "best_loss", "n_steps", "train_time")
        )
        rows.append(
            f"{name:<26}{runs[0]['n_params']:>8}"
            f"{rel.mean():>14.3e} +- {rel.std():<7.1e}"
            f"{jnp.median(best):>12.3e}{int(jnp.median(steps)):>8}"
            f"{jnp.median(time):>10.1f}"
        )
    return "\n".join(rows)


def load_results(run_dir: str | Path) -> Results:
    """Every model saved in a run folder."""
    return {
        p.stem: _load(p.with_suffix(""))
        for p in sorted((Path(run_dir) / "results").glob("*.json"))
    }


# ---------- Saving and loading ----------


def _save(stem: Path, settings, runs, u_exact, x) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: _pad([r[k] for r in runs]) for k in ARRAYS}
    if x is not None:
        arrays["x"] = jnp.asarray(x)
    jnp.savez(stem.with_suffix(".npz"), u_exact=jnp.asarray(u_exact), **arrays)
    meta = {
        "settings": settings,
        "runs": [{k: v for k, v in r.items() if k not in ARRAYS} for r in runs],
        "created": datetime.now().isoformat(timespec="seconds"),
        "versions": {p: version(p) for p in ("jax", "flax", "optax", "soap-jax")},
    }
    stem.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, default=lambda o: o.tolist())
    )


def _load(stem: Path) -> list[Run]:
    """The saved runs of one model."""
    meta = json.loads(stem.with_suffix(".json").read_text())
    with jnp.load(stem.with_suffix(".npz")) as data:
        runs = []
        for i, r in enumerate(meta["runs"]):
            n_checks = int(jnp.sum(~jnp.isnan(data["metric_steps"][i])))
            runs.append(
                {
                    **r,
                    "loss_history": jnp.asarray(
                        data["loss_history"][i, : r["n_steps"]]
                    ),
                    "rel_l2_history": jnp.asarray(data["rel_l2_history"][i, :n_checks]),
                    "metric_steps": jnp.asarray(
                        data["metric_steps"][i, :n_checks], dtype=int
                    ),
                    "u": jnp.asarray(data["u"][i]),
                }
            )
    return runs


# ---------- Array helpers ----------


def _pad(rows) -> Array:
    """Ragged per-seed curves as one (n_seeds, max_len) array, NaN-padded.

    Early stopping makes seeds stop at different steps; the plots use
    nan-aware reductions, so past a seed's end only the others count.
    """
    n = max(len(r) for r in rows)
    return jnp.stack(
        [
            jnp.pad(
                jnp.asarray(r, dtype=float), (0, n - len(r)), constant_values=jnp.nan
            )
            for r in rows
        ]
    )


def _common_steps(runs: list[Run]) -> tuple[Array, Array]:
    """Per-seed error curves interpolated onto one shared step axis.

    Early stopping ends each stage at a different step for each seed, so the
    i-th check of two seeds can be thousands of steps apart; stacking the
    curves by check index would misalign them.
    """
    steps = jnp.unique(jnp.concatenate([r["metric_steps"] for r in runs]))
    curves = jnp.stack(
        [
            jnp.interp(
                steps,
                r["metric_steps"],
                r["rel_l2_history"],
                left=jnp.nan,
                right=jnp.nan,
            )
            for r in runs
        ]
    )
    return steps, curves
