"""Error against degrees of freedom: the FEM sweeps next to the trained networks.
Reads saved results only, so nothing is retrained or re-solved.

Run 2D_poisson_L_shape.py (host) for each OMEGA_DEG and
2D_poisson_L_shape_fem.py (container) first, then:

    uv run experiments/2D_poisson_L_shape/plot_fem.py

For every corner angle with both, the FEM error is plotted against its dofs,
the networks' error is drawn as a level at their parameter count, and the dofs
each FEM method needs to match each network are written to a table.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

# plot_omega sits next to this file and puts the repository root on sys.path.
from plot_omega import (
    OUT_DIR,
    R_CORNER,
    RUNS,
    TITLE,
    collect,
    eval_points,
    load_problem,
)

from src.experiment import load_results
from src.metrics import rel_l2_error

# The networks compared with FEM, as keys of plot_omega.RUNS with the line
# style of their levels; the first is the baseline every omega needs.
NET_RUNS = {"grid": "-", "graded + bc": ":"}

# P1 and P2 by color, uniform and adaptive by line style.
FEM_STYLE = {
    "P1_uniform": dict(color="C0", ls="--", marker="o"),
    "P2_uniform": dict(color="C1", ls="--", marker="s"),
    "P1_adaptive": dict(color="C0", ls="-", marker="o"),
    "P2_adaptive": dict(color="C1", ls="-", marker="s"),
}
NET_COLOR = {"MLP": "C2", "KANN_cheb_decay": "C3"}


def load_fem(problem, omega_deg):
    """{method: arrays per refinement level}, errors scored like the networks.

    The npz also stores the error integrated over the mesh; here both errors
    are recomputed from the solution sampled on the networks' evaluation
    points, in the same precision as the networks.
    """
    pts = eval_points(problem, omega_deg, problem.N_EVAL)
    corner = jnp.linalg.norm(pts, axis=1) < R_CORNER
    u_exact = problem.exact_solution(pts[:, 0], pts[:, 1], omega_deg * jnp.pi / 180)

    fem = {}
    for method in FEM_STYLE:
        path = OUT_DIR / "runs" / f"fem-omega{omega_deg}" / "results" / f"{method}.npz"
        if not path.exists():
            continue
        with jnp.load(path) as data:
            d = {k: jnp.asarray(data[k]) for k in data.files}
        if not jnp.allclose(d["pts"], pts):
            raise ValueError(f"{path} was sampled on other points than the networks")
        d["rel_l2"] = jnp.array([rel_l2_error(u, u_exact) for u in d["u"]])
        d["corner_rel_l2"] = jnp.array(
            [rel_l2_error(u[corner], u_exact[corner]) for u in d["u"]]
        )
        fem[method] = d
    return fem


def dofs_to_reach(n_dofs, err, target):
    """Dofs at which err first drops to target, interpolated in log-log between
    levels. None if no level gets there; the first level's dofs if it already does."""
    below = jnp.nonzero(err <= target)[0]
    if len(below) == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(n_dofs[0])
    n0, n1 = jnp.log(n_dofs[i - 1 : i + 1].astype(float))
    e0, e1 = jnp.log(err[i - 1 : i + 1])
    return float(jnp.exp(n0 + (jnp.log(target) - e0) * (n1 - n0) / (e1 - e0)))


def plot(omega_deg, fem, nets, n_params, filename):
    """Global and corner error against dofs. A network is its median over seeds,
    with the seeds' range as a band, at its parameter count. nets and n_params
    are keyed by (model, run set), the run set one of NET_RUNS."""
    panels = (
        ("rel_l2", "rel. L2 error"),
        ("corner_rel_l2", f"rel. L2 error, r < {R_CORNER}"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.2), sharex=True)
    for ax, (key, label) in zip(axes, panels):
        for method, d in fem.items():
            ax.plot(
                d["n_dofs"],
                d[key],
                markersize=4,
                lw=1.5,
                label=f"FEM {method.replace('_', ' ')}",
                **FEM_STYLE[method],
            )
        for (model, points), m in nets.items():
            color = NET_COLOR.get(model, "C4")
            values = m[key]
            ax.axhline(
                jnp.median(values),
                color=color,
                ls=NET_RUNS[points],
                lw=1.5,
                label=f"{model} ({points})",
            )
            ax.axhspan(values.min(), values.max(), color=color, alpha=0.15, lw=0)
            ax.plot(
                n_params[model, points], jnp.median(values), "*", color=color, ms=12
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("degrees of freedom (networks: parameters, marked *)")
        ax.set_ylabel(label)
        ax.grid(True, which="major", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"{TITLE} omega {omega_deg} - FEM error against dofs, "
        f"lambda = {180 / omega_deg:.3f}"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary(rows) -> str:
    """Dofs each FEM method needs to match each network's median error."""
    methods = list(FEM_STYLE)
    header = f"{'omega':>6}  {'model':<32}{'params':>8}{'rel L2':>11}" + "".join(
        f"{m:>13}" for m in methods
    )
    lines = [
        "dofs for each FEM method to reach the network's median rel. L2 error",
        "",
        header,
        "-" * len(header),
    ]
    for omega_deg, model, params, err, dofs in rows:
        cells = "".join(
            f"{'-' if dofs.get(m) is None else f'{dofs[m]:.0f}':>13}" for m in methods
        )
        lines.append(f"{omega_deg:>6}  {model:<32}{params:>8}{err:>11.3e}{cells}")
    lines += [
        "",
        "-: the method never reached that error within its sweep.",
        "A value equal to the first level's dofs means the coarsest mesh already did.",
    ]
    return "\n".join(lines)


def main():
    problem = load_problem()
    tables = {points: collect(problem, RUNS[points]) for points in NET_RUNS}
    baseline = tables[next(iter(NET_RUNS))]
    figs = OUT_DIR / "figs"
    figs.mkdir(exist_ok=True)
    slug = TITLE.replace(" ", "_")

    rows = []
    omegas = sorted({w for by_omega in baseline.values() for w in by_omega})
    for omega_deg in omegas:
        fem = load_fem(problem, omega_deg)
        if not fem:
            continue
        nets, n_params = {}, {}
        for points, table in tables.items():
            run_dir = OUT_DIR / "runs" / f"{RUNS[points]}-omega{omega_deg}"
            results = load_results(run_dir)
            for model, by_omega in table.items():
                if omega_deg in by_omega:
                    nets[model, points] = by_omega[omega_deg]
                    n_params[model, points] = results[model][0]["n_params"]

        plot(
            omega_deg,
            fem,
            nets,
            n_params,
            str(figs / f"{slug}_fem_dofs_omega{omega_deg}.pdf"),
        )
        for (model, points), m in nets.items():
            target = float(jnp.median(m["rel_l2"]))
            dofs = {
                method: dofs_to_reach(d["n_dofs"], d["rel_l2"], target)
                for method, d in fem.items()
            }
            rows.append(
                (
                    omega_deg,
                    f"{model} ({points})",
                    n_params[model, points],
                    target,
                    dofs,
                )
            )

    text = summary(rows)
    print(text)
    (figs / f"{slug}_fem_dofs.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
