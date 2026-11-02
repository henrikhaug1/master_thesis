"""FEM reference for the 2D Laplace corner: error against degrees of freedom.

For every corner angle in OMEGA_DEGS, each method below is run over a sequence
of meshes, and every level is sampled on the networks' evaluation points so
plot_fem.py can score it exactly like the networks:

    P1_uniform, P2_uniform     uniform refinement of a coarse gmsh mesh
    P1_adaptive, P2_adaptive   residual estimator, Doerfler marking

Uniform refinement is limited by the corner singularity r^lambda to an L2 rate
of about N^-lambda whatever the degree; adaptive refinement recovers the
optimal rate N^-(p+1)/2.

Runs in the dolfinx container:

    docker compose exec fem python3 experiments/2D_poisson_L_shape/2D_poisson_L_shape_fem.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gmsh
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI

from src.fem import (
    doerfler_edges,
    eval_at,
    n_dofs,
    rel_l2_error,
    residual_estimator,
    save,
)

OUT_DIR = Path(__file__).parent

# ---------- Settings ----------
# Domain and evaluation grid as in 2D_poisson_L_shape.py, which cannot be
# imported here because it needs JAX.
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
N_EVAL = 128
OMEGA_DEGS = (180, 200, 225, 270, 300, 360)

H_COARSE = 1.0  # mesh size of the starting mesh, shared by every method
MAX_DOFS = 300_000  # a method stops once a level has more dofs than this
DOERFLER = 0.5  # adaptive: refine the fewest cells holding this share of eta^2
METHODS = {
    "P1_uniform": (1, False),
    "P2_uniform": (2, False),
    "P1_adaptive": (1, True),
    "P2_adaptive": (2, True),
}


# ---------- Problem ----------
def polar(x, y):
    """r and theta, with theta in [0, 2 pi) measured from the positive x-axis."""
    return np.sqrt(x**2 + y**2), np.mod(np.arctan2(y, x), 2 * np.pi)


def exact_solution(x, y, omega):
    lam = np.pi / omega
    r, theta = polar(x, y)
    # gmsh leaves nodes of the theta = 0 edge at y of about -1e-17, which
    # arctan2 puts at theta = 2 pi. Nothing in the domain lies beyond omega, so
    # such a theta can only be that edge.
    theta = np.where(theta > omega + 1e-6, theta - 2 * np.pi, theta)
    return r**lam * np.sin(lam * theta)


def exact_ufl(msh, omega):
    """The exact solution as a UFL expression, for the error integrated on the mesh.

    atan2 jumps on the negative x-axis, inside the domain; shifting it to
    [0, 2 pi) moves the jump to theta = 0, which is a boundary edge.
    """
    x = ufl.SpatialCoordinate(msh)
    lam = np.pi / omega
    atan = ufl.atan2(x[1], x[0])
    theta = ufl.conditional(ufl.lt(atan, 0), atan + 2 * np.pi, atan)
    return ufl.sqrt(x[0] ** 2 + x[1] ** 2) ** lam * ufl.sin(lam * theta)


def eval_points(omega):
    """The networks' evaluation points, in the order their predictions are saved."""
    X, Y = np.meshgrid(
        np.linspace(X_MIN, X_MAX, N_EVAL),
        np.linspace(Y_MIN, Y_MAX, N_EVAL),
        indexing="ij",
    )
    _, theta = polar(X, Y)
    mask = theta <= omega + 1e-6
    return np.stack([X[mask], Y[mask]], axis=-1)


# ---------- Mesh ----------
def coarse_mesh(omega):
    """The domain theta <= omega of the square, meshed by gmsh.

    The boundary runs from the corner out along theta = 0, around the square
    and back along theta = omega. At 360 degrees both ends land on (1, 0);
    they stay two separate gmsh points, so the slit gets two faces.
    """
    c, s = np.cos(omega), np.sin(omega)
    exit_r = 1.0 / max(abs(c), abs(s))
    # Snap the end of the theta = omega edge onto the square: at 225 degrees it
    # is the corner (-1, -1) only up to rounding, and the evaluation point there
    # would otherwise fall just outside the mesh. Coordinates near 0 are left
    # alone, so at 360 degrees the slit's far end stays a separate point from
    # (1, 0) rather than landing exactly on it, which gmsh cannot mesh.
    c, s = (
        float(np.sign(v)) if abs(abs(v) - 1.0) < 1e-9 else v
        for v in (exit_r * c, exit_r * s)
    )
    exit_r = 1.0
    square = [(X_MAX, Y_MAX), (X_MIN, Y_MAX), (X_MIN, Y_MIN), (X_MAX, Y_MIN)]
    # Strict inequality drops a square corner that the theta = omega edge ends on.
    outline = (
        [(0.0, 0.0), (X_MAX, 0.0)]
        + [p for p in square if polar(*p)[1] < omega - 1e-6]
        + [(exit_r * c, exit_r * s)]
    )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("corner")
    geo = gmsh.model.geo
    points = [geo.addPoint(x, y, 0.0, H_COARSE) for x, y in outline]
    lines = [geo.addLine(a, b) for a, b in zip(points, points[1:] + points[:1])]
    surface = geo.addPlaneSurface([geo.addCurveLoop(lines)])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], 1)
    gmsh.model.mesh.generate(2)
    msh = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2).mesh
    gmsh.finalize()
    return msh


# ---------- Solve ----------
def solve(msh, degree, omega):
    """Laplace with u = exact solution on the whole boundary."""
    V = fem.functionspace(msh, ("Lagrange", degree))
    g = fem.Function(V)
    g.interpolate(lambda x: exact_solution(x[0], x[1], omega))

    msh.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(msh.topology)
    bc = fem.dirichletbc(g, fem.locate_dofs_topological(V, 1, facets))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(msh, 0.0), v) * ufl.dx
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="corner_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    start = time.perf_counter()
    uh = problem.solve()
    return uh, time.perf_counter() - start


# ---------- Sweep ----------
def sweep(omega, degree, adaptive, pts):
    """One method from the coarse mesh until the dofs exceed MAX_DOFS."""
    msh = coarse_mesh(omega)
    levels = []
    while True:
        uh, solve_time = solve(msh, degree, omega)
        dofs = n_dofs(uh.function_space)
        levels.append(
            dict(
                n_dofs=dofs,
                n_cells=msh.topology.index_map(2).size_global,
                rel_l2_mesh=rel_l2_error(uh, exact_ufl(msh, omega)),
                solve_time=solve_time,
                u=eval_at(uh, pts),
            )
        )
        print(f"  {dofs:>8} dofs   rel L2 (mesh) {levels[-1]['rel_l2_mesh']:.3e}")
        if dofs > MAX_DOFS:
            return levels

        msh.topology.create_entities(1)
        edges = (
            doerfler_edges(msh, residual_estimator(uh), DOERFLER) if adaptive else None
        )
        msh, _, _ = mesh.refine(msh, edges)


def main():
    for omega_deg in OMEGA_DEGS:
        omega = omega_deg * np.pi / 180
        pts = eval_points(omega)
        for method, (degree, adaptive) in METHODS.items():
            print(f"omega {omega_deg}, {method}")
            levels = sweep(omega, degree, adaptive, pts)
            save(
                OUT_DIR
                / "runs"
                / f"fem-omega{omega_deg}"
                / "results"
                / f"{method}.npz",
                pts=pts,
                u_exact=exact_solution(pts[:, 0], pts[:, 1], omega),
                degree=np.array(degree),
                adaptive=np.array(adaptive),
                omega_deg=np.array(omega_deg),
                **{k: np.array([lvl[k] for lvl in levels]) for k in levels[0]},
            )


if __name__ == "__main__":
    main()
