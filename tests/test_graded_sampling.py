"""Checks that GradedCollocation places points and weights correctly on the
corner domains of experiments/2D_poisson_L_shape.

Run with `python tests/test_graded_sampling.py`, or with pytest if installed.
Run directly, it also saves a plot of the points to tests/test_plots.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import GradedCollocation, GridCollocation  # noqa: E402

# The experiment's file name starts with a digit, so it is loaded by path.
_spec = importlib.util.spec_from_file_location(
    "lshape", REPO_ROOT / "experiments/2D_poisson_L_shape/2D_poisson_L_shape.py"
)
lshape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lshape)

pi = jnp.pi
PLOT_DIR = Path(__file__).parent / "test_plots"
OMEGAS_DEG = (180, 200, 225, 270, 300, 360)
BETAS = (1.0, 2.0, 3.0)
GRADED = dict(n_r=48, n_theta=64)


def theta_integral(f, omega, n=200_000):
    """Midpoint rule for the integral of f(theta) over [0, omega]."""
    t = (jnp.arange(n) + 0.5) * omega / n
    return jnp.sum(f(t)) * omega / n


def exact_area(omega):
    """Area of the domain: the integral of R(theta)^2 / 2 dtheta."""
    return theta_integral(lambda t: lshape.square_radius(t) ** 2 / 2, omega)


def exact_grad2_integral(omega):
    """Integral of |grad u|^2 = lam^2 r^(2 lam - 2) over the domain, which is
    lam / 2 times the integral of R(theta)^(2 lam) dtheta. Singular at the
    corner for omega > 180 degrees."""
    lam = pi / omega
    return theta_integral(lambda t: lam / 2 * lshape.square_radius(t) ** (2 * lam), omega)


def graded(omega, beta):
    return GradedCollocation(**GRADED, beta=beta).points(lshape.domain(omega))


def test_points_inside_domain():
    for deg in OMEGAS_DEG:
        omega = deg * pi / 180
        for beta in BETAS:
            pts, _ = graded(omega, beta)
            x, y = pts[:, 0], pts[:, 1]
            assert pts.shape == (GRADED["n_r"] * GRADED["n_theta"], 2)
            assert jnp.all(lshape.in_domain(x, y, omega)), (deg, beta)
            assert jnp.all((jnp.abs(x) <= 1.0) & (jnp.abs(y) <= 1.0)), (deg, beta)
            assert jnp.all(x**2 + y**2 > 0.0), (deg, beta)  # never on the corner


def test_weights_sum_to_area():
    for deg in OMEGAS_DEG:
        omega = deg * pi / 180
        area = exact_area(omega)
        for beta in BETAS:
            _, w = graded(omega, beta)
            assert jnp.all(w > 0.0), (deg, beta)
            rel_err = abs(jnp.sum(w) - area) / area
            assert rel_err < 5e-3, (deg, beta, float(rel_err))


def test_unweighted_has_no_weights():
    omega = 270 * pi / 180
    _, w = GradedCollocation(**GRADED, weighted=False).points(lshape.domain(omega))
    assert w is None


def test_grading_beats_grid_on_corner_singularity():
    """With about as many points, graded points integrate the singular
    |grad u|^2 far more accurately than the uniform grid."""
    for deg in (225, 270, 300, 360):
        omega = deg * pi / 180
        lam = pi / omega
        exact = exact_grad2_integral(omega)

        pts, w = graded(omega, 2.0)
        r = jnp.linalg.norm(pts, axis=-1)
        graded_err = abs(jnp.sum(w * lam**2 * r ** (2 * lam - 2)) - exact) / exact

        # Grid points each stand for an equal share of the area.
        grid, _ = GridCollocation(n=64).points(lshape.domain(omega))
        r = jnp.linalg.norm(grid, axis=-1)
        w_grid = exact_area(omega) / grid.shape[0]
        grid_err = abs(jnp.sum(w_grid * lam**2 * r ** (2 * lam - 2)) - exact) / exact

        assert graded_err < 1e-3, (deg, float(graded_err))
        assert graded_err < grid_err / 10, (deg, float(graded_err), float(grid_err))


def plot_points(deg=270, beta=2.0):
    """The 64-grid next to the graded points, coloured by their area weights."""
    omega = deg * pi / 180
    grid, _ = GridCollocation(n=64).points(lshape.domain(omega))
    pts, w = graded(omega, beta)

    fig, (ax_grid, ax_graded) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    ax_grid.scatter(grid[:, 0], grid[:, 1], s=1)
    ax_grid.set_title(f"Grid 64 ({grid.shape[0]} points)")
    sc = ax_graded.scatter(pts[:, 0], pts[:, 1], s=1, c=jnp.log10(w))
    fig.colorbar(sc, ax=ax_graded, label="log10 area weight")
    ax_graded.set_title(f"Graded beta={beta:g} ({pts.shape[0]} points)")
    for ax in (ax_grid, ax_graded):
        ax.set_aspect("equal")
    fig.suptitle(f"Collocation points, omega = {deg} deg")

    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / f"graded_points_omega{deg}_b{beta:g}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"passed  {name}")
    print(f"All {len(tests)} tests passed.")
    print(f"saved   {plot_points().relative_to(REPO_ROOT)}")
