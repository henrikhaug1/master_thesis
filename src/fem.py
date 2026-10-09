"""Helpers shared by the DOLFINx reference solutions.

These run inside the dolfinx container, which has no JAX installed, so this
module stays pure numpy -- do not import anything from src.metrics here.
"""

from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, geometry
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
    """Relative L2 error, matching the metric src.metrics reports for the models."""
    msh = uh.function_space.mesh
    norm = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    return l2_error(uh, u_exact) / float(
        np.sqrt(msh.comm.allreduce(norm, op=MPI.SUM))
    )


def n_dofs(V: fem.FunctionSpace) -> int:
    """Global degrees of freedom, the FEM counterpart of src.metrics.n_params."""
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def save(path: str | Path, **arrays: np.ndarray) -> None:
    """np.savez into `path`, creating the results directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
