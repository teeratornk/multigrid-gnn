"""The removed Supplementary figures, redrawn from supplementary/figures/<name>/data.

Each function reads its inputs, returns the numbers the pre-cut Supplementary Information printed
about its figure, and draws the figure when given a path. matplotlib is imported only to draw, so
the numbers can be checked without it; install the `plots` extra to draw. The drawing follows the
scripts that made the published figures; the styling constants below are theirs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# The published figures' text convention: sans-serif with matching maths, TrueType fonts.
_BASE_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
               "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42}


def _pyplot(style: dict):
    import logging

    try:
        import matplotlib
    except ModuleNotFoundError as missing:
        raise ModuleNotFoundError("drawing the supplementary figures needs matplotlib; install the "
                                  "plots extra, for example uv sync --extra plots") from missing
    # Font subsetting in the PDF backend logs every table it prunes at INFO, which Hydra prints.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({**_BASE_STYLE, **style})
    return plt


def _data(root: Path, name: str) -> Path:
    return root / "figures" / name / "data"


def _exponent(ra: str) -> str:
    return ra.lower().split("e")[1]


# --- fig:cht-fields -------------------------------------------------------------------------------

def cht_fields(cfg, root: Path, pdf: Path | None = None) -> dict:
    """Isotherms and streamlines of the partitioned conjugate cavity across Ra."""
    fields = {ra: dict(np.load(_data(root, "cht-fields") / f"field_ra{ra}.npz")) for ra in cfg.ras}
    numbers = {f"nodes_ra{ra}": int(f["coords_p2"].shape[0]) for ra, f in fields.items()}
    numbers.update({f"temperature_range_ra{ra}": [float(f["T"].min()), float(f["T"].max())]
                    for ra, f in fields.items()})
    if pdf is None:
        return numbers
    plt = _pyplot({"font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 7,
                   "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6})
    import matplotlib.tri as mtri
    from scipy.interpolate import griddata

    tp = float(cfg.partition_thickness)
    x0, x1 = 0.5 - tp / 2, 0.5 + tp / 2
    levels = np.linspace(-0.5, 0.5, 25)
    lines = np.linspace(-0.5, 0.5, 13)
    fig, axes = plt.subplots(1, len(cfg.ras), figsize=(6.27, 2.14), constrained_layout=True)
    for ax, ra in zip(axes, cfg.ras):
        f = fields[ra]
        x, y = f["coords_p2"][:, 0], f["coords_p2"][:, 1]
        tri = mtri.Triangulation(x, y)
        centroid_x = x[tri.triangles].mean(axis=1)
        tri.set_mask((centroid_x > x0) & (centroid_x < x1))       # no fluid inside the partition
        ax.tricontourf(tri, f["T"], levels=levels, cmap="RdBu_r")
        ax.tricontour(tri, f["T"], levels=lines, colors="k", linewidths=0.35, alpha=0.55)
        n = int(cfg.streamline_grid)
        gx, gy = np.mgrid[0:1:n * 1j, 0:1:n * 1j]
        gu = griddata((x, y), f["u"], (gx, gy), method="linear")
        gv = griddata((x, y), f["v"], (gx, gy), method="linear")
        solid = (gx > x0) & (gx < x1)
        gu[solid] = np.nan
        gv[solid] = np.nan
        ax.streamplot(gx.T, gy.T, gu.T, gv.T, color="k", density=1.05, linewidth=0.6,
                      arrowsize=0.6)
        ax.add_patch(plt.Rectangle((x0, 0), tp, 1, facecolor="0.7", edgecolor="k", lw=0.8,
                                   zorder=5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(rf"$\mathrm{{Ra}}=10^{{{_exponent(ra)}}}$", fontsize=7)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.savefig(pdf, dpi=200)
    plt.close(fig)
    return numbers


# --- fig:cht-transient ----------------------------------------------------------------------------

def _trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["t"]) for r in rows]), np.array([float(r["Nu"]) for r in rows]))


def conduction_nu(kr: float, tp: float) -> float:
    """Steady composite-conduction Nu of the partitioned cavity: fluid width 1 - tp in series
    with a partition of thickness tp and conductivity ratio kr."""
    return 1.0 / ((1.0 - tp) + tp / kr)


def cht_transient(cfg, root: Path, pdf: Path | None = None) -> dict:
    """Hot-wall Nu against Khatamifar's convective time from the conduction start."""
    tp, prandtl = float(cfg.partition_thickness), float(cfg.prandtl)
    traces = {(kr, ra): _trace(_data(root, "cht-transient") / f"nu_kr{kr}_ra{ra}.csv")
              for kr in cfg.krs for ra in cfg.ras}
    numbers = {f"conduction_nu_kr{kr}": conduction_nu(float(kr), tp) for kr in cfg.krs}
    for (kr, ra), (t, nu) in traces.items():
        tau = np.sqrt(prandtl * float(ra)) * t
        numbers[f"kr{kr}_ra{ra}"] = dict(nu_start=float(nu[0]), nu_peak=float(nu.max()),
                                         nu_steady=float(nu[-1]), tau_end=float(tau[-1]),
                                         tau_peak=float(tau[int(np.argmax(nu))]))
    if pdf is None:
        return numbers
    plt = _pyplot({"font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 7,
                   "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6})
    colours = ("C0", "C1", "C2")
    fig, axes = plt.subplots(1, len(cfg.krs), figsize=(6.08, 2.32), sharey=False,
                             constrained_layout=True)
    for ax, kr in zip(axes, cfg.krs):
        for ra, colour in zip(cfg.ras, colours):
            t, nu = traces[(kr, ra)]
            tau = np.sqrt(prandtl * float(ra)) * t
            keep = tau > 0                             # the t=0 sample cannot sit on a log axis
            ax.plot(tau[keep], nu[keep], colour, lw=1.4,
                    label=rf"$\mathrm{{Ra}}=10^{{{_exponent(ra)}}}$")
            ax.axhline(nu[-1], color=colour, ls=":", lw=0.7, alpha=0.6)
        nuc = conduction_nu(float(kr), tp)
        ax.axhline(nuc, color="0.5", ls="--", lw=0.7)
        ax.text(560, nuc - 0.01, f"composite conduction {nuc:.2f}", color="0.4", fontsize=5.5,
                va="top", ha="right")
        ax.set_xscale("log")
        ax.set_xlim(1, 600)
        ax.set_xlabel(r"convective time $\tau=(\mathrm{Pr}\,\mathrm{Ra})^{1/2}\,t$")
        ax.set_title(cfg.titles[kr], fontsize=7)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel(r"hot-wall $\mathrm{Nu}(\tau)$")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6)
    fig.savefig(pdf, dpi=200)
    plt.close(fig)
    return numbers


# --- fig:engine-geom -------------------------------------------------------------------------------

def engine_geom(cfg, root: Path, pdf: Path | None = None) -> dict:
    """Not redrawn: the drawing script was not kept, and the mesh follows a commercial tutorial
    geometry that is not redistributed. The published figure is copied as it is."""
    if pdf is not None:
        pdf.write_bytes((root / "figures" / "engine-geom" / "published.pdf").read_bytes())
    return dict(redrawn=False)


# --- fig:rbc-stats ---------------------------------------------------------------------------------

def autocorrelation(x: np.ndarray) -> np.ndarray:
    """The normalised autocorrelation rho_k of a series, by FFT, for every lag k."""
    x = np.asarray(x, float)
    n = x.size
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n].real / (n - np.arange(n))
    return acov / acov[0]


def decorrelation_lag(rho: np.ndarray) -> int:
    """First zero crossing of the autocorrelation, in samples."""
    z = int(np.argmax(rho < 0.0))
    return z if z > 0 else rho.size


def blocking_curve(x: np.ndarray) -> list[tuple[int, float]]:
    """Flyvbjerg-Petersen blocking: (number of blocks, 95% interval) at each pairwise level."""
    xf = x - x.mean()
    rows = []
    while xf.size >= 8:
        rows.append((xf.size, 1.96 * xf.std(ddof=1) / np.sqrt(xf.size)))
        m = xf.size - xf.size % 2
        xf = 0.5 * (xf[0:m:2] + xf[1:m:2])
    return rows


def rbc_stats(cfg, root: Path, pdf: Path | None = None) -> dict:
    """Statistical convergence of the long Rayleigh-Bénard Nusselt trace."""
    with np.load(_data(root, "rbc-stats") / "nu_trace.npz") as d:
        t, nu_bot, nu_top = d["t"], d["nu_bot"], d["nu_top"]
        ra, prandtl = float(d["Ra"]), float(d["Pr"])
    turnovers = np.sqrt(ra * prandtl)               # free-fall times per thermal-diffusion time
    nu = 0.5 * (nu_bot + nu_top)
    dt = float(np.median(np.diff(t)))
    t_ff = t * turnovers
    developed = t >= float(cfg.warmup)
    x, tff_dev = nu[developed], t_ff[developed]
    rho = autocorrelation(x)
    tau_dec = decorrelation_lag(rho) * dt * turnovers
    curve = blocking_curve(x)
    reliable = [ci for blocks, ci in curve if blocks >= int(cfg.reliable_blocks)]
    ci = max(reliable) if reliable else curve[0][1]
    mean = float(np.mean(x))
    jd = float(cfg.johnston_doering.prefactor) * ra ** float(cfg.johnston_doering.exponent)
    numbers = dict(samples=int(t.size), span_turnovers=float(t_ff[-1] - t_ff[0]), mean_nu=mean,
                   ci95=float(ci), tau_dec_turnovers=float(tau_dec),
                   independent_samples=float((tff_dev[-1] - tff_dev[0]) / tau_dec),
                   johnston_doering_nu=jd, johnston_doering_excess_percent=100.0 * (mean - jd) / jd)
    if pdf is None:
        return numbers
    plt = _pyplot({"font.size": 6.0, "axes.labelsize": 6.5, "axes.titlesize": 7.0,
                   "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 5.5})
    running = np.cumsum(x) / np.arange(1, x.size + 1)
    fig, ax = plt.subplots(1, 3, figsize=(6.27, 2.35), constrained_layout=True)

    a = ax[0]
    a.plot(t_ff, nu, color="0.7", lw=0.3, label=r"$\mathrm{Nu}(t)$ (instantaneous)")
    a.plot(tff_dev, running, color="C0", lw=1.2, label="running mean (developed)")
    a.axhline(mean, color="C3", lw=0.9, label=rf"$\langle\mathrm{{Nu}}\rangle={mean:.2f}$")
    a.fill_between([tff_dev[0], t_ff[-1]], mean - ci, mean + ci, color="C3", alpha=0.18,
                   label="95% CI")
    a.axhline(jd, color="C2", ls="--", lw=0.9, label=rf"J&D 2009 DNS $={jd:.2f}$")
    a.axvline(float(cfg.warmup) * turnovers, color="0.4", ls=":", lw=0.7)
    a.text(float(cfg.warmup) * turnovers, a.get_ylim()[0] + 0.3, " warm-up", color="0.4",
           fontsize=5.5)
    a.set_xlabel("time (free-fall / turnover units)")
    a.set_ylabel(r"plate-averaged $\mathrm{Nu}$")
    a.set_ylim(0, max(20, nu.max() * 1.02))
    a.set_title(rf"RBC $\mathrm{{Ra}}=10^7$, $\Gamma=2$, $\mathrm{{Pr}}=0.71$ (h128)")
    a.legend(frameon=False, loc="lower right")
    a.grid(alpha=0.2)

    b = ax[1]
    lags = np.arange(rho.size) * dt * turnovers
    show = lags <= float(cfg.acf_max_lag)
    b.plot(lags[show], rho[show], color="C0", lw=1.0)
    b.axhline(0, color="0.5", lw=0.6)
    b.axvline(tau_dec, color="C3", ls="--", lw=0.9,
              label=rf"$\tau_{{\mathrm{{dec}}}}={tau_dec:.2f}$ turnovers")
    b.set_xlabel("lag (turnover units)")
    b.set_ylabel(r"autocorrelation $\rho(\mathrm{lag})$")
    b.set_title(rf"$\mathrm{{Nu}}$ autocorrelation: $N_{{\mathrm{{indep}}}}"
                rf"\approx{numbers['independent_samples']:.0f}$")
    b.legend(frameon=False)
    b.grid(alpha=0.2)

    c = ax[2]
    block_turnovers = [x.size / blocks * dt * turnovers for blocks, _ in curve]
    c.plot(block_turnovers, [v for _, v in curve], "o-", color="C0", ms=2.5, lw=0.9)
    for bt, (blocks, v) in zip(block_turnovers, curve):
        if blocks < int(cfg.reliable_blocks):
            c.plot(bt, v, "x", color="0.6", ms=4)
    c.axhline(ci, color="C3", ls="--", lw=0.9,
              label="largest retained\n" + rf"95% interval $=\pm{ci:.3f}$")
    c.set_xscale("log")
    c.set_xlabel("block length (turnover units)")
    c.set_ylabel("95% CI estimate")
    c.set_title("blocking-curve convergence")
    c.legend(frameon=False, loc="center right")
    c.grid(alpha=0.2, which="both")
    fig.savefig(pdf, dpi=200)
    plt.close(fig)
    return numbers


# --- fig:size-sweep --------------------------------------------------------------------------------

def _table(root: Path, name: str) -> list[dict]:
    with (root / "tables" / name / "data.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def size_sweep(cfg, root: Path, pdf: Path | None = None) -> dict:
    """Weak and strong scaling of the distributed solver, and the 2-to-8 GPU speedup against size.
    Drawn from the data of the removed tables weak, strong and strongmax, as the published figure
    was drawn from their printed values."""
    col = cfg.columns
    weak, strong, ceiling = (_table(root, cfg.tables[k]) for k in ("weak", "strong", "ceiling"))
    low, high = (int(g) for g in cfg.speedup_gpus)

    def series(rows, column):
        return {p: [float(r[column]) for r in rows if r[col.problem] == p] for p in cfg.problems}

    def at(rows, problem, gpus, column):
        return float(next(r[column] for r in rows
                          if r[col.problem] == problem and int(r[col.gpus]) == gpus))

    first = {p: min(int(r[col.gpus]) for r in ceiling if r[col.problem] == p) for p in cfg.problems}
    numbers = dict(
        weak_efficiency=series(weak, col.efficiency),
        strong_speedup=series(strong, col.speedup),
        problem_size={p: [at(strong, p, 1, col.total_dof), at(ceiling, p, first[p], col.total_dof)]
                      for p in cfg.problems},
        speedup_2_to_8={p: [at(strong, p, high, col.speedup) / at(strong, p, low, col.speedup),
                            at(ceiling, p, high, col.speedup) / at(ceiling, p, low, col.speedup)]
                        for p in cfg.problems},
    )
    if pdf is None:
        return numbers
    plt = _pyplot({"font.size": 6.5, "axes.linewidth": 0.6, "xtick.direction": "in",
                   "ytick.direction": "in"})
    from matplotlib.lines import Line2D

    styles = {p: dict(color=s.color, marker=s.marker) for p, s in cfg.styles.items()}
    gpus = [int(r[col.gpus]) for r in weak if r[col.problem] == cfg.problems[0]]
    fig, (aw, ast, az) = plt.subplots(1, 3, figsize=(6.27, 2.02), constrained_layout=True)
    mark = dict(lw=1.7, ms=6.5, mec="#2b2b2b", mew=0.5)

    aw.axhline(1.0, color="0.45", ls="--", lw=1.0)
    for p, y in numbers["weak_efficiency"].items():
        aw.plot(gpus, y, **mark, **styles[p])
    aw.set_xscale("log", base=2)
    aw.set_xticks(gpus)
    aw.set_xticklabels(gpus)
    aw.set_ylim(0, 1.08)
    aw.set_xlabel(r"GPUs  (fixed DOF/GPU, $\approx\!37$M)")
    aw.set_ylabel(r"weak efficiency  $t_1/t_N$")
    aw.set_title("(a) Weak scaling", fontsize=7)
    aw.text(0.97, 0.93, "ideal: flat", transform=aw.transAxes, ha="right", fontsize=5.5,
            color="0.45")

    ast.plot([1, 8], [1, 8], color="0.45", ls="--", lw=1.0)
    for p, y in numbers["strong_speedup"].items():
        ast.plot(gpus, y, **mark, **styles[p])
    ast.set_xscale("log", base=2)
    ast.set_yscale("log", base=2)
    ast.set_xticks(gpus)
    ast.set_xticklabels(gpus)
    ast.set_yticks([1, 2, 4, 8])
    ast.set_yticklabels([1, 2, 4, 8])
    ast.set_xlabel("GPUs")
    ast.set_ylabel(r"speedup  $T_1/T_N$")
    ast.set_title(r"(b) Strong scaling ($\approx\!54$M DOF)", fontsize=7)
    ast.text(0.04, 0.93, "ideal: linear", transform=ast.transAxes, fontsize=5.5, color="0.45")

    az.axhline(4.0, color="0.45", ls=":", lw=1.0)
    for p in cfg.problems:
        az.plot(numbers["problem_size"][p], numbers["speedup_2_to_8"][p], **mark, **styles[p])
    az.set_xlabel("problem size  [M DOF]")
    az.set_ylabel(r"$2\to8$ GPU speedup  $T_2/T_8$")
    az.set_title(r"(c) Scaling improves with size", fontsize=7)
    az.set_ylim(1.0, 4.25)
    az.text(0.97, 0.93, r"ideal: $4\times$", transform=az.transAxes, ha="right", fontsize=5.5,
            color="0.45")

    for axis in (aw, ast, az):
        axis.grid(alpha=0.22, which="both", lw=0.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(labelsize=6)
    fig.legend(handles=[Line2D([], [], label=p, **mark, **styles[p]) for p in cfg.problems],
               loc="lower center", ncol=len(cfg.problems), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.06))
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return numbers
