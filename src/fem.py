"""Helpers shared by the DOLFINx reference solutions.

These run inside the dolfinx container, which has no JAX installed, so this
module stays pure numpy -- do not import anything from src.metrics here.
"""

from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from mpi4py import MPI


def eval_at(uh: fem.Function, points: np.ndarray) -> np.ndarray:
    """Sample a finite element Function at arbitrary points.

    DOLFINx stores `uh` on degrees of freedom, not on a grid, so comparing it
    against the networks means locating the cell containing each point first.

    Args:
        uh: the solution Function.
        points: (N,) coordinates for a 1-D mesh, or (N, gdim) in general.

    Returns (N,) for a scalar space, (N, bs) otherwise, aligned with `points`.
    """
    pts = np.atleast_1d(np.asarray(points, dtype=np.float64))
    if pts.ndim == 1:
        pts = pts[:, None]

    # dolfinx always wants 3-D points, whatever the mesh dimension.
    padded = np.zeros((len(pts), 3), dtype=np.float64)
    padded[:, : pts.shape[1]] = pts

    msh = uh.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, padded)
    colliding = geometry.compute_colliding_cells(msh, candidates, padded)

    cells = np.empty(len(pts), dtype=np.int32)
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links) == 0:
            raise ValueError(f"point {pts[i]} lies outside the mesh")
        cells[i] = links[0]

    # eval drops the trailing axis for a single point, so reshape rather than index.
    values = np.asarray(uh.eval(padded, cells)).reshape(len(pts), -1)
    return values[:, 0] if values.shape[1] == 1 else values


def l2_error(uh: fem.Function, u_exact: ufl.core.expr.Expr) -> float:
    """||uh - u_exact||_L2, assembled on the mesh rather than sampled."""
    msh = uh.function_space.mesh
    e = uh - u_exact
    err = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    return float(np.sqrt(msh.comm.allreduce(err, op=MPI.SUM)))


def rel_l2_error(uh: fem.Function, u_exact: ufl.core.expr.Expr) -> float:
    """Relative L2 error integrated over the mesh.

    Not the networks' metric: src.metrics.rel_l2_error is a discrete norm over
    sampled points. Compare against the networks with that one, applied to the
    solution sampled with eval_at (as experiments/1D_poisson/plot.py does).
    """
    msh = uh.function_space.mesh
    norm = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    return l2_error(uh, u_exact) / float(np.sqrt(msh.comm.allreduce(norm, op=MPI.SUM)))


def residual_estimator(uh: fem.Function, f=0.0) -> np.ndarray:
    """Residual error estimate per cell for -Laplace u = f, as eta_K^2:

        h_K^2 ||f + Laplace u_h||_K^2 + sum over interior edges of h/2 ||[du_h/dn]||^2

    The Laplacian of u_h is zero for P1, leaving only f in the cell term.
    avg(w) hands each cell half of every edge it shares.
    """
    msh = uh.function_space.mesh
    W = fem.functionspace(msh, ("DG", 0))
    w = ufl.TestFunction(W)
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    eta2 = (
        h**2 * (f + ufl.div(ufl.grad(uh))) ** 2 * w * ufl.dx
        + ufl.avg(h) * ufl.jump(ufl.grad(uh), n) ** 2 * ufl.avg(w) * ufl.dS
    )
    return fem.assemble_vector(fem.form(eta2)).array


def doerfler_edges(msh: mesh.Mesh, eta2: np.ndarray, fraction: float) -> np.ndarray:
    """Edges of the fewest cells whose estimates sum to `fraction` of the total,
    for mesh.refine."""
    order = np.argsort(eta2)[::-1]
    n_marked = np.searchsorted(np.cumsum(eta2[order]), fraction * eta2.sum()) + 1
    msh.topology.create_connectivity(2, 1)
    return np.unique(
        mesh.compute_incident_entities(msh.topology, order[:n_marked], 2, 1)
    )


def n_dofs(V: fem.FunctionSpace) -> int:
    """Global degrees of freedom, the FEM counterpart of src.metrics.n_params."""
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def save(path: str | Path, **arrays: np.ndarray) -> None:
    """np.savez into `path`, creating the results directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
