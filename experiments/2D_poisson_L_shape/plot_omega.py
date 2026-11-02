"""Error against the corner angle omega, over every finished L-shape run, with
each set of collocation points in RUNS on the same axes. Reads saved results
only, so nothing is retrained.

Run 2D_poisson_L_shape.py once per OMEGA_DEG and COLLOCATION first, then:

    uv run experiments/2D_poisson_L_shape/plot_omega.py
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.experiment import load_results
from src.metrics import rel_l2_error

OUT_DIR = Path(__file__).parent
TITLE = "2D Laplace corner"
# Run folders are named f"{RUNS[label]}-omega{OMEGA_DEG}". The first entry is
# the baseline the others are compared with in the ratio table.
RUNS = {
    "grid": "soap6000-lbfgs4000",
    "graded": "soap6000-lbfgs4000-graded-b2",
    "graded + bc": "soap6000-lbfgs4000-graded-b2-bcb2",
}
LINESTYLES = ("-", "--", ":", "-.")  # one per entry of RUNS
R_CORNER = (
    0.1  # the corner error is measured on the evaluation points with r < R_CORNER
)


def load_problem():
    """The experiment script as a module, so the domain and exact solution used
    here are the ones the networks were trained and scored on."""
    path = OUT_DIR / "2D_poisson_L_shape.py"
    spec = importlib.util.spec_from_file_location("lshape", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_points(problem, omega_deg, n_eval):
    """The evaluation points of one run, in the order its predictions are saved."""
    omega = omega_deg * jnp.pi / 180
    g_x = jnp.linspace(problem.X_MIN, problem.X_MAX, n_eval)
    g_y = jnp.linspace(problem.Y_MIN, problem.Y_MAX, n_eval)
    X, Y = jnp.meshgrid(g_x, g_y, indexing="ij")
    mask = problem.in_domain(X, Y, omega)
    return jnp.stack([X[mask], Y[mask]], axis=-1)


def collect(problem, run):
    """{model: {omega_deg: metrics}} over the run folders named f"{run}-omega*",
    each metric an array with one entry per seed."""
    table = {}
    for run_dir in OUT_DIR.glob(f"runs/{run}-omega*"):
        results = load_results(run_dir)
        if not results:
            continue  # a run still in progress has no results yet
        name = next(iter(results))
        meta = json.loads((run_dir / "results" / f"{name}.json").read_text())
        omega_deg = meta["settings"]["problem"]["omega_deg"]
        pts = eval_points(problem, omega_deg, meta["settings"]["problem"]["n_eval"])
        u_exact = problem.exact_solution(pts[:, 0], pts[:, 1], omega_deg * jnp.pi / 180)
        corner = jnp.linalg.norm(pts, axis=1) < R_CORNER

        for model, runs in results.items():
            table.setdefault(model, {})[omega_deg] = {
                "rel_l2": jnp.array([r["rel_l2"] for r in runs]),
                "corner_rel_l2": jnp.array(
                    [rel_l2_error(r["u"][corner], u_exact[corner]) for r in runs]
                ),
                "max_err": jnp.array([r["max_err"] for r in runs]),
                # Runs whose last stage ran out of steps rather than stopping itself.
                "capped": sum(r["stop_reasons"][-1] == "max_steps" for r in runs),
                "n_seeds": len(runs),
            }
    return {m: dict(sorted(by_omega.items())) for m, by_omega in table.items()}


def _degrees_to_lambda(v):
    """lambda = 180 / omega_deg, which is also its own inverse. Matplotlib probes
    the axis at 0 while laying it out, hence the floor."""
    return 180 / jnp.maximum(jnp.asarray(v, dtype=float), 1e-9)


def plot(tables, filename):
    """Global and corner error against omega: median over seeds, every seed as a
    dot. One colour per model, one line style per set of collocation points."""
    panels = (
        ("rel_l2", "rel. L2 error"),
        ("corner_rel_l2", f"rel. L2 error, r < {R_CORNER}"),
    )
    models = sorted({m for table in tables.values() for m in table})
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {m: cycle[i % len(cycle)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2))
    for ax, (key, label) in zip(axes, panels):
        for (points, table), style in zip(tables.items(), LINESTYLES):
            for model, by_omega in table.items():
                omegas = jnp.array(list(by_omega))
                values = [m[key] for m in by_omega.values()]
                ax.plot(
                    omegas,
                    [jnp.median(v) for v in values],
                    marker="o",
                    linestyle=style,
                    color=colors[model],
                    label=f"{model} ({points})",
                )
                for w, v in zip(omegas, values):
                    ax.scatter(
                        jnp.full(v.shape, w), v, s=12, alpha=0.4, color=colors[model]
                    )
        ax.set_xlabel("omega [deg]")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        # lambda = pi / omega is what sets the singularity, r^lambda near the corner.
        top = ax.secondary_xaxis("top", functions=(_degrees_to_lambda,) * 2)
        top.set_xlabel("lambda = pi / omega")
    axes[0].legend()
    fig.suptitle(f"{TITLE} - error against the corner angle (median over seeds)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary(table) -> str:
    """One row per model and omega; the ratio compares each model with the MLP."""
    header = (
        f"{'omega':>6}{'lambda':>8}  {'model':<20}{'rel L2 (mean +- std)':>24}"
        f"{'corner rel L2':>16}{'max err':>12}{'/ MLP':>8}{'capped':>8}"
    )
    rows = [header, "-" * len(header)]
    omegas = sorted({w for by_omega in table.values() for w in by_omega})
    for w in omegas:
        for model, by_omega in table.items():
            if w not in by_omega:
                continue
            m = by_omega[w]
            rel = m["rel_l2"]
            mlp = table.get("MLP", {}).get(w)
            ratio = f"{jnp.mean(rel) / jnp.mean(mlp['rel_l2']):>8.2f}" if mlp else ""
            rows.append(
                f"{w:>6}{180 / w:>8.3f}  {model:<20}"
                f"{rel.mean():>14.3e} +- {rel.std(ddof=1):<7.1e}"
                f"{jnp.median(m['corner_rel_l2']):>16.3e}"
                f"{jnp.median(m['max_err']):>12.3e}{ratio:>8}"
                f"{m['capped']:>5}/{m['n_seeds']}"
            )
        rows.append("")
    rows.append(
        "capped: seeds whose last stage hit its step limit instead of stopping itself."
    )
    return "\n".join(rows)


def ratio_summary(tables) -> str:
    """Each set of points against the first in RUNS, per omega and model: the
    errors of both and their ratio; below 1 means the other set did better."""
    (base, base_table), *others = tables.items()
    blocks = []
    for points, table in others:
        # Columns wide enough for the labels, which name the sets of points.
        wb = max(16, len(base) + 9)
        wo = max(18, len(points) + 9)
        header = (
            f"{'omega':>6}  {'model':<20}"
            f"{'rel L2 ' + base:>{wb}}{'rel L2 ' + points:>{wo}}{'ratio':>8}"
            f"{'corner ' + base:>{wb}}{'corner ' + points:>{wo}}{'ratio':>8}"
        )
        rows = [f"{points} / {base}", header, "-" * len(header)]
        omegas = sorted(
            {w for m in table.values() for w in m}
            & {w for m in base_table.values() for w in m}
        )
        for w in omegas:
            for model in sorted(table):
                if w not in table[model] or w not in base_table.get(model, {}):
                    continue
                # Means for the global error and medians for the corner error,
                # as in summary().
                rel_b = jnp.mean(base_table[model][w]["rel_l2"])
                rel_o = jnp.mean(table[model][w]["rel_l2"])
                cor_b = jnp.median(base_table[model][w]["corner_rel_l2"])
                cor_o = jnp.median(table[model][w]["corner_rel_l2"])
                rows.append(
                    f"{w:>6}  {model:<20}"
                    f"{rel_b:>{wb}.3e}{rel_o:>{wo}.3e}{rel_o / rel_b:>8.2f}"
                    f"{cor_b:>{wb}.3e}{cor_o:>{wo}.3e}{cor_o / cor_b:>8.2f}"
                )
            rows.append("")
        blocks.append("\n".join(rows))
    return "\n".join(blocks)


def main():
    problem = load_problem()
    tables = {points: collect(problem, run) for points, run in RUNS.items()}
    tables = {points: table for points, table in tables.items() if table}
    figs = OUT_DIR / "figs"
    figs.mkdir(exist_ok=True)
    slug = TITLE.replace(" ", "_")

    plot(tables, str(figs / f"{slug}_omega_sweep.pdf"))
    sections = [f"== {points} ({RUNS[points]})\n{summary(t)}" for points, t in tables.items()]
    if len(tables) > 1:
        sections.append(f"== ratios\n{ratio_summary(tables)}")
    text = "\n\n".join(sections)
    print(text)
    (figs / f"{slug}_omega_sweep.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
