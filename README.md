# gnn4buoyancy-mwe: a multigrid V-cycle as message-passing layers

Repository `multigrid-gnn`, distribution `gnn4buoyancy-mwe`, import `gnn4buoyancy`.

Minimal working example accompanying *Classical finite-element solvers as differentiable
graph programs on GPUs*.

It contains one component of that work: the fixed-weight algebraic multigrid pressure
solve, written as message passing, small enough to read in full and to run on a laptop.
It is not a reproduction of the benchmark. The production solvers are proprietary and are
not released. Reproduction of the reported numbers rests on the specification in Methods
and in the Supplementary reproducibility note, together with the per-run configurations,
which will be deposited on publication.

Version 1.1.0 adds six pressure-solve experiments that run from one command each. They run
on this package's own operators, and every number they print is a number about those
operators. None of them reproduces a value reported for the production solvers.

Version 1.2.0 is the first public release, under the MIT licence. It adds `supplementary/`: the
tables, figures and text moved out of the paper's Supplementary Information, kept as data with the
code that rebuilds the tables and four of the five figures and checks them against what was
printed. Unlike the rest of the package, these are results of the production solvers. See
`supplementary/README.md`.

## Contents

| path | what it holds |
|---|---|
| `src/gnn4buoyancy/layers.py` | the sparse apply, as gather, weight, scatter-add |
| `src/gnn4buoyancy/hierarchy.py` | smoothed aggregation frozen into fixed edge weights |
| `src/gnn4buoyancy/vcycle.py` | the V-cycle, as a sequence of message-passing layers |
| `src/gnn4buoyancy/pcg.py` | conjugate gradients preconditioned by one V-cycle |
| `src/gnn4buoyancy/cases.py` | the three example operators |
| `src/gnn4buoyancy/experiments.py` | the experiment app, one function per experiment |
| `src/gnn4buoyancy/reference.py` | the V-cycle restated in scipy, the reference for the identity |
| `src/gnn4buoyancy/stationary.py` | the V-cycle used as a standalone stationary solver |
| `src/gnn4buoyancy/diagnostics.py` | the measurements the experiments record |
| `src/gnn4buoyancy/summary.py` | run summaries, and their comparison with expectations |
| `src/gnn4buoyancy/supplementary.py` | the app for the material moved out of the Supplementary Information |
| `src/gnn4buoyancy/figures.py` | four of the removed Supplementary figures, redrawn from their inputs |
| `src/gnn4buoyancy/conf/` | the Hydra configuration groups |
| `experiments/expected/` | the committed expectation for each experiment and profile |
| `supplementary/` | the removed tables, prose-only results, figure inputs and the pre-cut Supplementary pages |
| `tests/` | the suite described below |

## The construction

A sparse operator application is message passing on the degree-of-freedom graph:

```
(A x)_i = sum_{j in N(i)} A_ij x_j
```

This is an exact identity. A row of a sparse matrix is a message-passing
neighbourhood whose edge weights are the matrix entries. A multigrid V-cycle, meaning
smooth, restrict, recurse, prolong, smooth, is therefore a sequence of message-passing
layers on a hierarchy of graphs, with the weights taken from the assembled finite-element
operator. Nothing is trained. `tests/test_message_passing.py::test_nothing_is_trainable`
asserts that no tensor in the model carries a gradient or is a parameter.

## Requirements

Python 3.10, 3.11 or 3.12. The dependencies are torch, numpy, scipy, pyamg and
hydra-core; drawing the supplementary figures also needs matplotlib, in the `dev` and `plots`
extras. `uv.lock` pins the exact versions the results below were produced with. No
GPU is required: every case here is small by design and runs on a laptop CPU. Only the
`paper` experiment profile asks for a GPU.

## Install and run

```bash
uv sync --locked --extra dev
uv run gnn4buoyancy                                    # conjugate cavity, graph multigrid
uv run gnn4buoyancy case=heat_sink                     # conjugate fin array, k_s/k_f = 1000
uv run gnn4buoyancy case=engine_bore                   # engine mid-plane, gas bore in metal
uv run gnn4buoyancy case.params.n=128 solver.omega=0.5
uv run gnn4buoyancy solver=jacobi_ref                  # single level, for comparison
```

`uv sync --locked` installs exactly the versions in `uv.lock`, and fails instead of
resolving new ones. On Linux the locked torch is the CUDA build from PyPI, so the
environment takes about 4.7 GB, of which 2.7 GB is NVIDIA runtime. On a machine with no
route to a package index, sync once where there is one, then use
`uv run --frozen --offline ...`, which neither resolves nor downloads.

A laptop can skip the NVIDIA runtime with
`uv venv && uv pip install -e ".[dev]" --torch-backend=cpu`. That install does not read
the lock, so its versions are not the tested ones.

### Expected output

The first command prints the following, up to timings and the last digits of the residual:

```
case=conjugate_cavity  n=4096  nnz=20224
hierarchy: 4 levels, sizes [4096, 704, 80, 9], setup 0.011s
solve: 10 iterations, relative residual 7.090e-09, converged, 0.009s
wrote result.json
```

The last digits move because smoothed aggregation draws a start vector from NumPy's global
random state, which the CLI does not seed. The experiments below seed it.

`solver=jacobi_ref` solves the same system with the same conjugate gradients and no
hierarchy. It takes 115 iterations against 10.

`dtype: float32` is available but will not reach the default `rtol: 1.0e-8`, which sits
below single-precision epsilon. It reports `converged: false` and the true residual it
reached. The residual is recomputed as `b - Ax` before convergence is declared, because the
CG recurrence drifts. Use a looser `solver.rtol` in float32.

## Experiments

Six experiments run the pressure solve on the three example operators. Each runs from one
command and writes a summary. Every number in this section describes these small operators.
None of them is a value from the production solvers, and none stands in for one.

| profile | grids | device | in the default test run |
|---|---|---|---|
| `quick` | 16 x 16 and 32 x 32 | CPU | yes, about 5 s for all six |
| `paper` | 32 x 32, 64 x 64 and 128 x 128 | one GPU | only where CUDA is available |

```bash
uv run gnn4buoyancy-experiment experiment=vcycle_identity profile=quick
uv run gnn4buoyancy-experiment experiment=vcycle_spd profile=quick
uv run gnn4buoyancy-experiment experiment=hierarchy_rebuild profile=quick
uv run gnn4buoyancy-experiment experiment=stationary_vs_pcg profile=quick
uv run gnn4buoyancy-experiment experiment=stationary_mismatch profile=quick
uv run gnn4buoyancy-experiment experiment=mixed_precision profile=quick
```

Use `profile=paper` for the GPU grids. Hydra's `-m` runs several in one call, for example
`uv run gnn4buoyancy-experiment -m profile=quick experiment=vcycle_identity,vcycle_spd`.

Every experiment uses the V-cycle of `solver/graph_mg.yaml`: weighted Jacobi with omega 0.7,
two sweeps before and two after, the coarsest level relaxed, and CG to a relative residual
of 1e-8. NumPy is seeded before every hierarchy build and torch runs its deterministic
kernels, so a rerun on the same machine gives the same summary bit for bit.

What each experiment shows, with the ranges recorded in `experiments/expected/`:

| experiment | what it shows | quick, CPU | paper, GPU |
|---|---|---|---|
| `vcycle_identity` | The graph V-cycle equals the same cycle written in scipy, run on the hierarchy the graph holds, for 3 cases and 9 smoother settings. | relative difference 3.5e-18 to 8.8e-17 | 1.7e-17 to 2.1e-16 |
| `vcycle_spd` | R equals P^T exactly on every level. The cycle is symmetric at 2 and 2 sweeps and visibly unsymmetric at 3 and 1. `<x,Bx>` is positive. | symmetry defect at most 1.8e-17 at 2 and 2, against 2.1e-4 to 3.5e-3 at 3 and 1 | at most 2.0e-17 at 2 and 2, against 5.2e-5 to 2.5e-3 at 3 and 1 |
| `hierarchy_rebuild` | Two builds from one NumPy seed are equal bit for bit. Other seeds move the coarse operators, and no iteration count changes. | coarse drift 2.3e-4 to 8.1e-3 | 1.0e-4 to 1.6e-2, solutions agree to 1.3e-10 |
| `stationary_vs_pcg` | The cycle as a stationary solver takes more iterations than the same cycle inside CG, at omega 1 and at the best fixed omega. Its rate is the one the extreme eigenvalues of BA predict. The count does not grow steadily with the grid. | 1.25 to 32 times the CG count | 1.5 to 13.9 times; heat_sink at omega 1 takes 123, 279 and 77 against 16, 20 and 17 |
| `stationary_mismatch` | A cycle built on the uniform operator and iterated on the contrast 1e3 operators diverges at omega 1, 0.5, 0.3, 0.1 and 0.05, because lambda_max(BS) is about 700 and 1000. At 0.9 of 2/lambda_max it does not diverge. CG with the same cycle converges. | diverges within 6 iterations; CG takes 58 to 102 | diverges within 6 iterations; CG takes 89 to 170 |
| `mixed_precision` | CG in fp64 with the fp32 cycle takes the fp64 iteration count. The whole solve in fp32 does not reach 1e-8. | 0 extra iterations; fp32 stops at 1.4e-7 or above | 0 extra; fp32 stops at 2.6e-7 to 3.9e-4 |

The two stationary experiments show a mechanism on these operators. The Supplementary
note's iteration counts, and their growth with refinement, belong to the production
pressure operator, which is not part of this package.

Each run writes `<name>.summary.json` into its Hydra run directory under `outputs/`. It holds
the rows measured, the claims evaluated on them, and the environment: package, Python, torch
and CUDA versions, device and dtype. `mode=check` compares the run with
`experiments/expected/<name>.<profile>.json` and exits nonzero on any disagreement.
`mode=refresh` rewrites that file. Expectation files hold no versions, timings, hosts or
paths. The tolerance for each field sits under `regression:` in the experiment's config,
beside the spread it was set from, and a field without a rule must match exactly.
`tests/test_mutations.py` breaks each claim on purpose and shows that the check then fails.

## Supplementary material

`supplementary/` holds the tables, figures and text moved out of the paper's Supplementary
Information. `supplementary/README.md` describes what is there and what is checked.

```bash
uv run gnn4buoyancy-supplementary                 # tables as Markdown and LaTeX, figures as PDF
uv run gnn4buoyancy-supplementary mode=check      # checks only, exits nonzero on any mismatch
```

## Configuration

Every run parameter is a Hydra config group under `src/gnn4buoyancy/conf/`: `case/` selects
the operator, `solver/` selects the preconditioner, and `experiment/` holds one file per
experiment. These are this example's own groups. The longer list in the paper's
Supplementary reproducibility note belongs to the production solver, which is not part of
this package. Nothing is hardcoded in the source, which `tests/test_config.py` checks by
grepping for numeric literals outside signature defaults. Each result file records the
resolved configuration that produced it.

The experiment app composes `conf/experiments.yaml`. It takes the V-cycle from
`solver/graph_mg.yaml` and the operators from `case/*.yaml`, the same files the CLI reads,
so an experiment runs the configuration a user runs.

One combination is refused. `solver.n_pre` and `solver.n_post` are independent keys, but a
V-cycle is a symmetric preconditioner only when they are equal, and conjugate gradients is
valid only for a symmetric preconditioner. Setting them unequal raises an error. `v_cycle`
itself accepts unequal counts and is tested with them. Only the conjugate gradient wrapper
refuses them.

## Tests

```bash
uv run pytest -q                 # full suite
uv run pytest -q -m "not slow"   # quick subset
```

It checks, among others:

| what is checked | where |
|---|---|
| the message-passing identity, row by row | `test_message_passing.py` |
| nothing in the model is trainable | `test_message_passing.py` |
| restriction is the transpose of prolongation | `test_hierarchy.py` |
| the graph V-cycle reproduces a numpy reference of the same cycle | `test_vcycle.py` |
| the V-cycle is symmetric and positive definite, so CG is valid | `test_vcycle.py` |
| the solution solves the system, against a direct sparse solve | `test_solver.py` |
| the hierarchy beats a single level and is near mesh independent | `test_solver.py` |
| unequal smoothing counts are refused by the CG wrapper | `test_solver.py` |
| the reported residual is the true one, not a drifted recurrence | `test_solver.py` |
| float32 tracks float64, and GPU matches CPU | `test_dtype_device.py` |
| the stationary solver matches a numpy loop at every iterate | `test_diagnostics.py` |
| Ritz values match the eigenvalues of the dense product | `test_diagnostics.py` |
| two builds from the same NumPy seed are equal bit for bit | `test_diagnostics.py` |
| `pcg` without the new keywords is the previous code, bit for bit | `test_diagnostics.py` |
| the version agrees in pyproject, the package, CITATION.cff and uv.lock | `test_diagnostics.py` |
| the licence agrees in LICENSE, pyproject and CITATION.cff | `test_diagnostics.py` |
| a removed table's data is its published LaTeX, digit for digit | `test_supplementary.py` |
| every exported file matches its digest, and nothing unlisted is present | `test_supplementary.py` |
| the printed numbers listed for each removed figure are recomputed from its inputs | `test_supplementary.py` |
| the quick profile's claims hold and match their expectations | `test_experiments.py` |
| the paper profile matches its expectations on a GPU | `test_experiments.py` |
| no run parameter is hardcoded outside a signature default | `test_config.py` |
| the wheel builds and carries the Hydra config groups | `test_packaging.py` |

The V-cycle identity test is the one that bears on the paper's central claim. Its
reference is a numpy implementation of the same cycle on the same hierarchy, with the same
relaxation and sweep counts and the coarsest level smoothed rather than solved. PyAMG's
own preconditioner is deliberately not the reference: it defaults to Gauss-Seidel and
solves the coarsest level directly, so disagreement with it would say nothing about
whether this execution is faithful.

## Verification status

276 tests, of which some depend on the machine rather than on the code:

| environment | result |
|---|---|
| NVIDIA H200 | **276 pass, 0 skip** |
| CPU only | 268 pass, 8 skip (both parametrizations of `test_gpu_matches_cpu`, and `test_paper_on_gpu_matches_expected` for each experiment) |
| no build backend available to uv | the wheel-build test skips, since it cannot build |

At the tagged commit, with a clean working tree, the full suite passes on an H200 and every
experiment passes in check mode for both profiles. Version 1.1.0 has the same solver and
experiment code. A fresh clone of its tag, installed with `uv sync --locked --extra dev`, gave
the same result, and its paper summaries were identical bit for bit to those of a working-tree
run on another node.

`run_gpu_check.sbatch` reproduces the GPU leg of the test suite, and
`run_experiments.sbatch` runs every experiment, the quick profile on the CPU and the paper
profile on the GPU, against its expectation. Set `EXPERIMENTS_MODE=refresh` to rewrite the
expectations instead. Both scripts record the commit and the number of modified files next
to the result, so a log can be tied to the tree that produced it. Both run
`uv run --frozen --offline`, so a job uses the locked environment or fails. Neither sets
an account, partition or queue, since those are site names that mean nothing elsewhere.
Pass whatever your site needs on the command line, where it overrides the directives in
the file:

```bash
sbatch --account=<acct> --qos=<qos> run_gpu_check.sbatch
sbatch --account=<acct> --qos=<qos> run_experiments.sbatch
```

The wheel-build test shells out to a real build, and the `cht-fields` drawing test grids three
fields; both are marked slow. `pytest -m "not slow"` skips those two and nothing else.

## The three cases

| case | interface | why it is included |
|---|---|---|
| `conjugate_cavity` | none | uniform coefficients, the base case |
| `heat_sink` | straight | a conductivity jump on flat fin boundaries |
| `engine_bore` | curved | the jump falls on a staircased boundary, which is hardest for the coarsening |

`engine_bore` is a parametric mid-plane section: a gas bore inside a liner inside a metal
block. It is not the geometry on which the paper's engine results were computed,
and that mesh is not part of this example. It is included because a curved solid-fluid
interface is the feature that makes the applied case demanding for smoothed aggregation,
and neither other case has one.

## Scope

Not included: the engine mesh, the distributed multi-GPU solver, the conjugate
flow coupling, and the differentiability study. This example is the pressure solve, which
is the component the construction is named for.

Smoothed aggregation is not bit-reproducible across rebuilds unless NumPy is seeded first.
PyAMG estimates a spectral radius from a random start vector drawn from NumPy's global
state, so two unseeded builds of the same matrix differ in their coarse operators, by 1e-4
to 2e-2 relative on these cases. Each build is still a valid preconditioner, and no
iteration count here changes. Seed NumPy before the build, as the experiments do, and a
rebuild is identical bit for bit.

## Licence and citation

Released under the MIT licence; see `LICENSE`. The production solvers this example comes from
are proprietary and are not part of this repository. `CITATION.cff` carries the citation
metadata for this package and for the paper it accompanies: T. Kadeethum et al., *Classical
finite-element solvers as differentiable graph programs on GPUs*.
