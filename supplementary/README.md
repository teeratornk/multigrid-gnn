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
- Adjoint comparison: the tape memory at 94291 DOF is 65.2 GiB. The pre-cut table printed 65.2 GiB,
  but the pre-cut text gave 65.3 GiB, which is the value of the M=8 run.
- Stopping criterion: every value of the table now comes from one rerun. At h_256 and Ra=10^4 the
  GNN value is 2.2446 with a change of 0.014, not 2.2445 and 0.018, and the FEniCSx value is 2.2366
  under both tests, not 2.2367. At Ra=10^5 the GNN change is 0.052, not 0.053.
- Pressure backends: on the 2D cavity, mixed precision is 61 to 84% slower per step, not 60 to 84%.
- Rayleigh-Bénard statistics: the offset of the mean Nusselt number from the Johnston-Doering
  correlation, 0.40, is more than ten times the ±0.025 interval, not two orders of magnitude larger.
- Hardware and software: the GPU runs used Python 3.12, PyTorch 2.5.1+cu124, CUDA 12.4 and NCCL
  2.21.5, not Python 3.11, PyTorch 2.2.2+cu118, CUDA 11.8 and NCCL 2.19.3.

## Errors found in this record after release

A later audit checked most GNN results in this record against the run records, and reran runs whose
settings differ from what the text states. The timing reruns of the 3D cavity size sweep,
`notes/rbc-block-jacobi-h724`, `notes/engine-matched-tolerance`, `tables/mgsolver` and
`tables/strongscale-n161` each ran next to a control run at the published settings, and differ from
it only in the settings in question. The reruns of `notes/partition-thickness` and
`figures/cht-transient` used a later version of the driver with all the stated pressure settings.
The files above still keep every value as it was printed.

### Wrong values

- `tables/engine-mesh`: the coarse liner-side mean is 394.4 K, not 394.5 K.
- `tables/strongscale-n161`: the Nicolaides time per PCG iteration on eight GPUs is 4.06 ms, the
  median of five runs, not 4.07 ms, the median of three.
- `tables/weak-transient`:
  - On one GPU the conjugate cavity needs 80 CG iterations per step with the two-level preconditioner
    and 81 with block-Jacobi, not 241 and 242. The count was already per step and was multiplied by
    three a second time. The strong-scaling table of the Supplementary Information prints the same
    runs as 80 and 81.
  - At the pressure cap, block-Jacobi is 1.8 to 2.4 times slower than two-level on the conjugate
    cavity, not 1.9 to 2.5. The 2.5 is the two-GPU run, which is below the cap.
- `tables/mms-distributed`:
  - The pressure order at h_128 is 2.04, not 2.05. The orders use a mesh ratio of 2.
  - At h_64 the two preconditioners agree to 1.6×10^-9 in velocity and 2.2×10^-8 in pressure, not to
    about 10^-9.
  - The single-GPU runs match the four-GPU runs to 4×10^-10 up to h_64 with Nicolaides deflation, and
    to 2.2×10^-8 with block-Jacobi, not to about 10^-9 at every mesh. No single-GPU run existed at
    h_128. One made for this audit matches to 7×10^-8 with Nicolaides deflation and 6.6×10^-7 with
    block-Jacobi. All values agree at the printed digits.
- `tables/mgsolver`:
  - At Ra=10^6 preconditioned CG needs 5600 SIMPLE iterations and the standalone solver 5200. The
    table prints 5200 for both.
  - Preconditioned CG gives Nu 8.8467 at Ra=10^6 on h_64 and 2.2452 at Ra=10^4 on h_128. The table
    prints the standalone values, 8.8466 and 2.2451, as serving both solvers.
  - The operator match is 0.15 at h_64 and 0.14 at h_128, not 0.15 at both.
  - The wall ratio is a ratio of solve times. End to end it is 2.3, 2.9, 2.5 and 17.3.
- `tables/strongmax`: the 3D cavity h_144 mesh file has 63.4 M DOF in total and 31.7, 15.8 and 7.9 M
  per GPU, not 63.7 M and 31.8, 15.9 and 8.0 M, which scale a reference count. The published runs read
  a file of the same name that is not available, so their count is not confirmed. The 3D cavity DOF in
  `tables/strong` and `tables/weak` are scaled estimates as well.
- `notes/mms-modal-coefficients`:
  - The cos(2πx)cos(2πy) mode carries over 99.8% of the squared pressure error at h_256 and h_512, not
    at h_256 and coarser. It carries 57.5% at h_64 and 27.0% at h_128.
  - The slopes 2.75 for temperature and 1.76 for pressure are not least-squares fits over the full 2D
    mesh range. They come from the two finest meshes alone, measured against the minimum mesh spacing,
    which shrinks 2.2 times between h_256 and h_512. Against the nominal halving the same pair gives
    3.1 and 2.0, and a least-squares fit over all four meshes gives 3.96 and 2.21.
- `notes/partition-thickness`: at thickness 0.2 and Ra=10^6 the finest mesh gives Nu 2.122, not
  2.123. The reductions stay 3.7 and 18.2%. The third is 24.0% from the unrounded values and 24.1%
  from the printed ones, so it sits on a rounding boundary.
- `notes/engine-matched-tolerance`: matching both codes at 10^-7 moves the ratios by 0.47% and 0.51%
  at the two finer meshes and by 4.3% at the coarsest, not by at most 0.5% and 3.7%. Measured within
  one job, the tolerance mismatch inflates the ratio 1.25 to 1.26 times, not 1.25 to 1.28.
- `notes/scaling-captions`:
  - `tab:strongscale`: the block-Jacobi kernel speedup on eight GPUs is 2.03, not 2.04.
  - `tab:weakscale`: the efficiency of 0.65 at the lighter load comes from 2.0 to 3.0 ms per PCG
    iteration, not 4.5 to 6.9 ms. Those came from an earlier outer-iteration metric that was
    withdrawn. Deflation costs 7 to 8% more per iteration, not about 10%.
  - `tab:weak-ceiling`: the engine's peak allocation of 104 GB is 69% of the memory of one H200, not
    73%.
- `figures/cht-transient`: the dotted lines mark the last logged Nusselt number of each run, not the
  steady values of the main text. The two differ by at most 0.003.

### Runs that differ from what the text states

- 3D cavity rows of `tables/strong`, `tables/strongmax`, `tables/weak` and `figures/size-sweep`:
  - On two, four and eight GPUs the pressure solves sit at the solver's default cap of 100 iterations,
    while one GPU needs 24 to 56 on average, so these entries time capped work. The configuration table
    gives a cap of 500 for the distributed-cavity scaling sweeps, and no caption says these runs used
    100. The captions and the section text also leave out Ra=10^5 and the 300 outer iterations from
    rest, and nothing says that ± is a population standard deviation.
  - In a rerun on the h_144 mesh at that cap of 500, the pressure needs 169 to 263 iterations per
    outer step, and the speedups on two, four and eight GPUs are 0.88, 1.16 and 1.24, not 1.15, 1.83
    and 2.44. These are medians of three over outer iterations 2 to 300, because iteration 1 sometimes
    stalls. With the cap at 100 the same window gives 1.14, 1.85 and 2.52.
  - The meshes of the other 3D cavity rows are not available, and the mesh generator's output depends
    on a thread count that was not recorded, so those rows were not rerun.
- `figures/size-sweep`: the caption says the speedup rises with mesh size for every problem. That
  holds for the plotted pairs. The unplotted Rayleigh-Bénard h_2048 mesh, which is larger than h_1980,
  gives a two-to-eight-GPU speedup of 1.56 against 1.71 on h_1980.
- `notes/rbc-block-jacobi-h724`: the multi-GPU runs stop at 100 pressure iterations per solve, where
  the protocol states 300. At 300 the speedups on two, four and eight GPUs are 0.27, 0.17 and 0.14
  and the parallel efficiencies 0.13, 0.04 and 0.02, not 0.36 to 0.40 and 0.18 to 0.05. These are
  medians of three. On eight GPUs even the cap of 300 is almost reached. With the cap at 100 the
  rerun reproduces the note. Block-Jacobi still does not strong-scale here.
- `tables/mgsolver`: the standalone multigrid ran without CUDA-graph capture and the preconditioned
  CG with it, so the settings were not identical. With neither captured, the standalone solver over CG
  is 0.88, 1.71 and 1.46 at h_64 and 7.5 at h_128, not 2.3, 3.0, 2.5 and 17.4. These are medians of
  three runs of each solver on a node other than the published one. With capture as published, the
  same runs give 2.5, 3.1, 2.5 and 16.8. The inner iteration counts agree within 1% as medians. That
  the preconditioned CG ran captured follows from its configuration; its logs do not record it.
- `tables/strongscale-n161` and `notes/coarse-space-ablation`:
  - The runs used the weighted Jacobi smoother and no deferred correction, and Nicolaides deflation
    applied one fine V-cycle per rank. The configuration table gives the Chebyshev smoother, with
    weighted Jacobi only as an option, deferred correction on the cavity runs, and five fine multigrid
    iterations per rank for Nicolaides deflation.
  - `tables/strongscale-n161` was rerun with the Chebyshev smoother and deferred correction, still
    with one fine V-cycle, because the code applies one V-cycle whatever count is configured.
    Iterations change by at most 4%, the time per PCG iteration rises by 0.8 to 5.1%, and the speedups
    over iterations 2 to 50 move by at most 0.014, so its conclusions hold.
  - Its speedup column averages 50 outer iterations, including a first iteration that costs up to
    17 s. Over iterations 2 to 50 the same runs give 1.48 and 1.74 for block-Jacobi and 1.53 and 1.93
    for Nicolaides, not 1.42, 1.47, 1.42 and 1.62.
  - Its within-cell spread below 5% holds for the time per PCG iteration, but not for the time per
    outer iteration behind the speedup column, which spreads 11 to 24% over the five runs.
  - The ablation calls its affine cells deterministic, but each is a single run, so their
    repeatability was not measured.
- `notes/scaling-captions`, `tab:strongscale`: its speedups and walls also average 50 outer
  iterations, including a first iteration that takes about 1.6 s on one GPU and up to 17 s on two to
  eight. From iteration 2 on, block-Jacobi reaches 1.83 on both four and eight GPUs, so it flattens
  rather than regressing, and Nicolaides deflation reaches 2.16 on four and 2.34 on eight, so it does
  not turn over. The eight-GPU walls are 549 against 702 ms, not 750 against 829.
- `tables/mms-distributed`: the wall-clock columns are single runs of 5000 unconverged iterations,
  without setup, with two four-GPU runs started together on each node.
- `tables/msiter`, GNN column: each value is measured within the first 500 to 1200 outer iterations
  from rest, which the caption does not say.
- `tables/weak-transient`: both one-GPU rows come from the strong-scaling runs.
- `notes/engine-matched-tolerance`: the GPU runs 200 steps and the CPU 60, not the same number.
  Each mean includes the first step, which on the GPU costs 2 to 15 later steps. In a rerun with
  both implementations in one job, on two exclusive nodes with three repetitions each, running the
  GPU for 60 steps as well lowers the ratios by 1 to 6% against the published pairing, and the
  steady cost per step from step 5 on changes them by -3 to +2%. The published pairing reproduces
  the printed ratios to within 10% on both nodes. On a third node, where the first GPU step stayed
  slow, the 60-step GPU mean was 7 to 13% above the 200-step one. At matched steps the fire-deck
  peaks agree to four digits.
- `notes/partition-thickness`, `figures/cht-transient` and `figures/cht-fields`: the pressure took five
  fixed PCG iterations preconditioned by a weighted-Jacobi V-cycle, not the stated tolerance of
  10^-6, cap of 50 and Chebyshev smoother. Rerun at the stated settings, the Nu values for thickness
  0.2 agree to 10^-6 and the transient traces to a relative 8.4×10^-5, with the same peaks.
  `figures/cht-fields` was not rerun.
- `notes/scaling-captions`: the 3.45 onset ratio of `tab:weak-ceiling` is a single run.

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
