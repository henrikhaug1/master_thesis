"""Thesis figures for 1D Poisson: the trained networks of one run next to the FEM
reference. Reads saved results only, so nothing is retrained.

Run 1Dpoisson.py (host) and 1Dpoisson_fem.py (container) first, then:

    uv run experiments/1D_poisson/plot.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from src.experiment import _common_steps, _pad, load_results, median_prediction
from src.metrics import max_error, rel_l2_error
from src.plotting import plot_loss_bands, plot_solution_grid

OUT_DIR = Path(__file__).parent
TITLE = "1D Poisson"
RUN = "soap6000-lbfgs4000"  # folder under runs/ holding the networks
FEM_FILE = OUT_DIR / "runs" / "fem" / "results" / "fem.npz"


def load_fem(path):
    """The FEM solution, its error measured exactly like the networks' error.

    fem.npz also stores the error integrated over the mesh, but the networks
    are scored on the evaluation points, so both are recomputed here from the
    sampled solution, in float32 like the networks.
    """
    with jnp.load(path) as data:
        fem = {k: jnp.asarray(data[k]) for k in data.files}
    fem["rel_l2"] = rel_l2_error(fem["u"], fem["u_exact"])
    fem["max_err"] = max_error(fem["u"], fem["u_exact"])
    fem["label"] = f"FEM P{int(fem['degree'])}, {int(fem['n_dofs'])} dofs"
    return fem


def summary(results, fem) -> str:
    """The run's table with an FEM row; dofs go in the params column."""
    header = (
        f"{'model':<26}{'params':>8}{'rel L2 (mean +- std)':>24}"
        f"{'max err':>12}{'time [s]':>10}"
    )
    rows = [header, "-" * len(header)]
    for name, runs in results.items():
        rel, err, time = (
            jnp.array([r[k] for r in runs]) for k in ("rel_l2", "max_err", "train_time")
        )
        rows.append(
            f"{name:<26}{runs[0]['n_params']:>8}"
            f"{rel.mean():>14.3e} +- {rel.std():<7.1e}"
            f"{jnp.median(err):>12.3e}{jnp.median(time):>10.1f}"
        )
    solve_time = (
        f"{float(fem['solve_time']):>10.3f}" if "solve_time" in fem else f"{'-':>10}"
    )
    rows.append(
        f"{'FEM':<26}{int(fem['n_dofs']):>8}{fem['rel_l2']:>14.3e}{'':<11}"
        f"{fem['max_err']:>12.3e}{solve_time}"
    )
    return "\n".join(rows)


def main():
    results = load_results(OUT_DIR / "runs" / RUN)
    fem = load_fem(FEM_FILE)
    x = fem["x"]
    figs = OUT_DIR / "figs"
    figs.mkdir(exist_ok=True)
    slug = TITLE.replace(" ", "_")

    # FEM has no training history, so it only appears as a reference level.
    curves = {n: _common_steps(runs) for n, runs in results.items()}
    plot_loss_bands(
        {n: c for n, (_, c) in curves.items()},
        str(figs / f"{slug}_rel_l2.pdf"),
        f"{TITLE} - relative L2 error (median over seeds)",
        steps={n: s for n, (s, _) in curves.items()},
        y_label="rel. L2 error",
        references={fem["label"]: fem["rel_l2"]},
    )
    plot_loss_bands(
        {n: _pad([r["loss_history"] for r in runs]) for n, runs in results.items()},
        str(figs / f"{slug}_loss.pdf"),
        f"{TITLE} - loss (median over seeds)",
    )

    plot_solution_grid(
        x,
        fem["u_exact"],
        {
            **{n: median_prediction(runs) for n, runs in results.items()},
            fem["label"]: fem["u"],
        },
        "x",
        "u(x)",
        TITLE,
        str(figs / f"{slug}_solutions.pdf"),
    )

    table = summary(results, fem)
    print(table)
    (figs / "summary.txt").write_text(table + "\n")


if __name__ == "__main__":
    main()
