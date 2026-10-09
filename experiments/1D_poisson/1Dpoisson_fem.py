import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from src.fem import eval_at, n_dofs, rel_l2_error, save

X_MIN, X_MAX = 0.0, 1.0
N_POINTS = 200

N_ELEMENTS = 64
DEGREE = 2

OUT_DIR = Path(__file__).parent


def main():
    msh = mesh.create_interval(MPI.COMM_WORLD, N_ELEMENTS, [X_MIN, X_MAX])
    V = fem.functionspace(msh, ("Lagrange", DEGREE))

    facets = mesh.locate_entities_boundary(msh, 0, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological(V, 0, facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.pi**2 * ufl.sin(ufl.pi * x[0])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    start = time.perf_counter()
    uh = problem.solve()
    solve_time = time.perf_counter() - start

    xq = np.linspace(X_MIN, X_MAX, N_POINTS)
    u_fem = eval_at(uh, xq)
    u_exact = np.sin(np.pi * xq)

    rel_l2 = rel_l2_error(uh, ufl.sin(ufl.pi * x[0]))
    max_err = float(np.max(np.abs(u_fem - u_exact)))
    dofs_total = n_dofs(V)

    save(
        OUT_DIR / "runs" / "fem" / "results" / "fem.npz",
        x=xq,
        u=u_fem,
        u_exact=u_exact,
        rel_l2=np.array(rel_l2),
        max_err=np.array(max_err),
        n_dofs=np.array(dofs_total),
        n_elements=np.array(N_ELEMENTS),
        degree=np.array(DEGREE),
        solve_time=np.array(solve_time),
    )

    print(f"P{DEGREE} on {N_ELEMENTS} elements, {dofs_total} dofs")
    print(f"rel L2 {rel_l2:.3e}   max err {max_err:.3e}   solve {solve_time:.3f} s")


if __name__ == "__main__":
    main()
