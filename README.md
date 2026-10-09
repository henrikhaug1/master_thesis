# master_thesis

PINNs (MLP and KANN variants) for PDEs, with FEM reference solutions.

## Running the experiments

The networks run on the host, in the uv environment:

```bash
uv run experiments/1D_poisson/1Dpoisson.py
```

## Changing an experiment

Every setting sits in a block at the top of the experiment script: the problem
(grid size, coefficients, hard or soft boundary conditions), `TRAIN` and
`MODELS`. The options of each are documented in `src/config.py`.

```python
TRAIN = TrainConfig(
    stages=(Stage("soap", 6000, lr=3e-3), Stage("lbfgs", 4000)),
    seeds=(0, 1, 2),
)
```

Each `Stage` is one optimizer (`"adam"`, `"adamw"`, `"sgd"`, `"soap"`,
`"lbfgs"`), run for up to `steps` steps, starting where the previous stage
stopped. Adam only is `(Stage("adam", 10000, lr=1e-3),)`, L-BFGS only
`(Stage("lbfgs", 5000),)`. Further optimizer arguments go in `options`.

```python
MODELS = {
    "MLP": MLPConfig((1, 48, 48, 48, 1), activation="tanh", init="glorot_normal"),
    "KANN_cheb": KANNConfig((1, 16, 16, 16, 1), basis="cheb",
                            basis_options=dict(degree=5, decay=-2.0), coeff_std=0.05),
}
```

## Results

Each run is saved in its own folder next to the script,
`runs/<run name>/`, where the run name defaults to the optimizer stages
(e.g. `soap6000-lbfgs4000`):

```
runs/soap6000-lbfgs4000/
    results/<model>.npz    loss and error histories, predictions, exact solution
    results/<model>.json   all settings, per-seed errors, steps, training time,
                           stop reasons and package versions
    summary.txt            the results table
    figs/*.pdf
```

Every run trains all models from scratch and overwrites the run folder's
results. Pass `run_name=...` to `compare_models` to keep a variant in its own
folder. `src.experiment.load_results(run_dir)` reads a finished run back.

## Running the FEM reference solutions

The FEM code needs DOLFINx, which is not in the uv environment. It runs in the
`dolfinx/dolfinx:stable` container instead. Start the container once:

```bash
docker compose up -d
```

then run scripts in it:

```bash
docker compose exec fem python3 experiments/1D_poisson/1Dpoisson_fem.py
```

For an interactive shell in the same environment:

```bash
docker compose exec fem bash
```

Stop it with `docker compose down` when you are finished. A single run without
starting the service works too: `docker compose run --rm fem python3 ...`.

The repository is mounted at `/work`, so files written by the container (for
example `runs/fem/results/fem.npz`) appear directly in the working tree.

The two environments are deliberately disjoint -- the container has no JAX, and
the host has no DOLFINx. `src/fem.py` is therefore pure numpy; importing
`src.metrics` or anything else JAX-backed from it would break the container.
