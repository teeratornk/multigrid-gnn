# Material moved out of the paper's Supplementary Information

The paper's Supplementary Information was shortened from 57 to 29 pages before submission. What
left it is kept here as data, with the code that turns the data back into the printed tables and
four of the five figures, and checks them against what was printed.

These are results of the production solvers, which are not part of this package. The package does
not recompute them. It keeps them so they can be read, reused and checked.

## What is here

| path | what it holds |
|---|---|
| `si_before_cut.pdf` | the Supplementary pages before the cut, with every removed table, figure and paragraph in its original place; one table row is rebuilt without site scheduler flags |
| `tables/<name>/published.tex` | a removed table, as printed |
| `tables/<name>/data.csv` | its values, one row per printed row |
| `notes/<name>/published.tex` | a paragraph or caption whose results were printed only there |
| `notes/<name>/data.csv` | its values, kept by hand and checked against the paragraph |
| `figures/<name>/published.pdf` | a removed figure, as printed |
| `figures/<name>/data/` | the inputs the figure was drawn from |
| `manifest.yaml` | what each exported file is, and a digest of its content |

The files were exported from the paper's source repository. No file name or field carries a host,
a job, a path or a branch.

In `data.csv`, a table printed in groups has a leading `group` column. A cell the table prints
once for several rows is repeated on each of them.

Each table's caption, note and cross-references, such as `sec:si-scaling-protocol`, are in its
`published.tex`. The rendered Markdown tables carry the caption and the note. A cross-reference
points into `si_before_cut.pdf`.

## Tables

| name | table |
|---|---|
| `cnn-probes` | Robustness probes for the CNN velocity and pressure floor |
| `engine-mesh` | Engine meshes and metal-temperature QoI, grid convergence |
| `mgsolver` | Standalone multigrid versus multigrid-preconditioned conjugate gradients |
| `mms-distributed` | Multi-GPU manufactured-solution check, as printed before the cut with the two wall-clock columns the cut removed |
| `msiter` | Per-SIMPLE-iteration cost, 2D and 3D benchmarks |
| `strong` | Strong scaling at matched problem size |
| `strongmax` | Strong scaling at the largest meshes |
| `strongscale-n161` | Strong scaling on a larger fixed problem |
| `weak` | Weak scaling at fixed total DOF per GPU |
| `weak-transient` | Weak scaling of the two transient 2D problems at the single-GPU memory ceiling |

## Notes

Some results that left the Supplementary Information were printed in a paragraph or a caption, not
in a table. Each is kept as printed, in `notes/<name>/published.tex`, with its values in
`notes/<name>/data.csv`. The values were written by hand from the paragraph, and the check confirms
that every number of every value is printed in it.

| name | results |
|---|---|
| `rbc-block-jacobi-h724` | block-Jacobi strong scaling of the unsteady Rayleigh-Bénard cell on the h_724 mesh |
| `coarse-space-ablation` | pressure iterations with an affine coarse space, against block-Jacobi and the indicator space |
| `partition-thickness` | conjugate cavity Nusselt numbers at partition thicknesses 0.1 and 0.2 |
| `engine-matched-tolerance` | engine throughput ratios with both codes matched at a tolerance of 1e-7 |
| `scaling-captions` | scaling results printed only in the pre-cut captions of three tables the paper keeps |
| `mms-modal-coefficients` | modal coefficients of the GNN pressure error in the 2D manufactured solution |
| `cnn-ramp-coefficients` | the linear pressure ramp in the CNN manufactured-solution error |

## Figures

| name | figure | drawn from |
|---|---|---|
| `cht-fields` | Flow structure in the conjugate cavity | steady velocity and temperature on the P2 nodes, at three Rayleigh numbers |
| `cht-transient` | Transient development of the hot-wall Nusselt number | hot-wall Nusselt number against time, from six runs |
| `rbc-stats` | Statistical convergence of the turbulent Rayleigh-Bénard Nusselt number | the plate Nusselt traces of one long run |
| `size-sweep` | Scaling of the distributed solver against problem size | the tables `weak`, `strong` and `strongmax` |
| `engine-geom` | Engine conjugate geometry and mesh | not redrawn |

`engine-geom` is kept as the published figure only. The script that drew it was not kept, and its
mesh follows a commercial tutorial geometry that is not redistributed here.

## Corrections made after the cut

The pre-cut record keeps every value as it was printed. After the cut, the paper's Supplementary
Information corrected the values below, each against the run records. The paper holds the corrected
values.

- Rayleigh-Bénard block-Jacobi time per step: 1154 ms on one GPU and 2032 ms on two, not 1155 and
  2033 ms. The old 1155 is also in `tables/weak-transient`.
- Weak scaling at the single-GPU memory ceiling: the DOF were counted exactly. Every DOF entry of
  that table changed except the engine on one GPU, and no time or efficiency changed. On eight GPUs
  the heat sink has 1.18×10^8 DOF, not 1.127×10^8, and the engine 4.67×10^8, not 4.94×10^8. On one
  GPU the heat sink has 1.57×10^7, not 1.52×10^7. The old heat sink values, 1.13×10^8 and its
  rounding 1.1×10^8, are also in `notes/scaling-captions`.
- Heat sink grid convergence: the fine mesh has 6.95×10^6 DOF and the finest 1.57×10^7, not
  6.82×10^6 and 1.52×10^7.
- Two-level strong scaling: the conjugate cavity iteration counts are per step, not per solve. Those
  runs cap the pressure at 100 iterations per solve, not 300.
- Timing protocol: the heat sink scaling runs are single runs. The pre-cut protocol made every timing
  a median of three unless its caption said otherwise.
- CPU time from 16 to 32 cores: the engine range is -3 to +1%, not -1 to +3%, and the pyadjoint
  range is -3 to +6%, not -5 to +3%. The old engine range is also in `notes/engine-matched-tolerance`.
- Adjoint comparison: the tape memory at 94291 DOF is 65.3 GiB, not 65.2 GiB.
- Stopping criterion: every value now comes from one rerun. At h_256 and Ra=10^4 the GNN value is
  2.2446 with a change of 0.014, not 2.2445 and 0.018, and the FEniCSx value is 2.2366 under both
  tests, not 2.2367. At Ra=10^5 the GNN change is 0.052, not 0.053.
- Pressure backends: mixed precision is 61 to 84% slower per step in 2D, not 60 to 84%.
- Rayleigh-Bénard statistics: the offset of the mean Nusselt number from the Johnston-Doering
  correlation, 0.40, is 16 times the ±0.025 interval, not two orders of magnitude larger.
- Hardware and software: the GPU runs used Python 3.12, PyTorch 2.5.1+cu124, CUDA 12.4 and NCCL
  2.21.5, not Python 3.11, PyTorch 2.2.2+cu118, CUDA 11.8 and NCCL 2.19.3.

## Run

```bash
uv run gnn4buoyancy-supplementary                 # tables as Markdown and LaTeX, figures as PDF
uv run gnn4buoyancy-supplementary mode=check      # checks only, exits nonzero on any mismatch
```

Run from the repository root. Each run writes to `outputs/supplementary/<date>_<time>/`. Drawing
the figures needs matplotlib, which `uv sync --locked --extra dev` or `--extra plots` installs.
Without it the default mode still writes the tables and runs the checks, then exits nonzero with a
message. `mode=check` does not need matplotlib. `mode=refresh` rewrites each `data.csv` from its
`published.tex`.

## What is checked

`mode=check` and `tests/test_supplementary.py` hold this material to the printed record:

- each `data.csv` is what its `published.tex` parses to;
- the numbers of each printed row, read from the LaTeX itself, are the numbers of its data row;
- every number of every value in a note's `data.csv` is printed in its `published.tex`;
- every file matches its digest in `manifest.yaml`, and no unlisted file is present;
- the numbers the pre-cut captions and text printed about the figures, listed under `printed:` in
  `src/gnn4buoyancy/conf/supplementary.yaml`, are recomputed from the figures' inputs and match at
  the printed digits. Examples are the Rayleigh-Bénard mean Nusselt number with its interval, and
  the start and peak Nusselt numbers of the transient runs. The steady transient values are not
  checked, because the text printed grid-converged values from other runs. `cht-fields` has no
  printed numbers.

The tests also break each of these, in a copy of the files, in the configuration or in the LaTeX
conversion, and show that the check then fails.
