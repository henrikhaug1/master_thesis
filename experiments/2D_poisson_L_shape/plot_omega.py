"""Error against the corner angle omega, over every finished L-shape run. Reads
saved results only, so nothing is retrained.

Run 2D_poisson_L_shape.py once per OMEGA_DEG first, then:

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
RUN = "soap6000-lbfgs4000"  # run folders are named f"{RUN}-omega{OMEGA_DEG}"
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


def collect(problem):
    """{model: {omega_deg: metrics}}, each metric an array with one entry per seed."""
    table = {}
    for run_dir in OUT_DIR.glob(f"runs/{RUN}-omega*"):
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


def plot(table, filename):
    """Global and corner error against omega: median over seeds, every seed as a dot."""
    panels = (
        ("rel_l2", "rel. L2 error"),
        ("corner_rel_l2", f"rel. L2 error, r < {R_CORNER}"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2))
    for ax, (key, label) in zip(axes, panels):
        for model, by_omega in table.items():
            omegas = jnp.array(list(by_omega))
            values = [m[key] for m in by_omega.values()]
            (line,) = ax.plot(
                omegas, [jnp.median(v) for v in values], marker="o", label=model
            )
            for w, v in zip(omegas, values):
                ax.scatter(
                    jnp.full(v.shape, w), v, s=12, alpha=0.4, color=line.get_color()
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


def main():
    table = collect(load_problem())
    figs = OUT_DIR / "figs"
    figs.mkdir(exist_ok=True)
    slug = TITLE.replace(" ", "_")

    plot(table, str(figs / f"{slug}_omega_sweep.pdf"))
    text = summary(table)
    print(text)
    (figs / f"{slug}_omega_sweep.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
