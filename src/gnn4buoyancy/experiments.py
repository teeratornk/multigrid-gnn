"""Hydra app: `gnn4buoyancy-experiment experiment=stationary_mismatch profile=quick`.

Six pressure-solve experiments on the package's own operators. Each sweeps its grid inside
one run, because its claim is a relation across that grid, and returns a summary: rows of
measurements, the claims evaluated on them, and the environment they ran in. `run(cfg)`
does the work and writes nothing. `main` writes `<run_dir>/<name>.summary.json` and, in
`check` or `refresh` mode, compares it with or rewrites the committed expectation.

NumPy is seeded before every hierarchy build. pyamg's smoothed aggregation estimates a
spectral radius from NumPy's global RNG, which is what made two builds of the same matrix
differ. Random vectors come from a CPU torch.Generator, so every device sees the same ones.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from . import summary
from .cases import build_case
from .diagnostics import (bitwise_equal, coarse_drift, convergence_factor, min_rayleigh,
                          relative_difference, ritz_values, stationary_radius,
                          symmetry_defect, transpose_defect)
from .hierarchy import build_hierarchy, cast_levels
from .layers import MessagePassingOperator
from .pcg import pcg
from .reference import reference_vcycle
from .stationary import stationary
from .vcycle import v_cycle


def _sizes(cfg: DictConfig) -> list[int]:
    return list(cfg.experiment.profiles[cfg.profile].sizes)


def _matrix(cfg: DictConfig, case: str, n: int):
    params = OmegaConf.to_container(cfg.cases[case].params, resolve=True)
    return build_case(cfg.cases[case].name, **{**params, "n": n})


def _levels(cfg: DictConfig, A, *, seed: int | None = None):
    np.random.seed(cfg.seed if seed is None else seed)
    return build_hierarchy(A, max_levels=cfg.vcycle.max_levels, dtype=getattr(torch, cfg.dtype),
                           device=cfg.device, use_sparse_mm=cfg.vcycle.use_sparse_mm)


def _vectors(cfg: DictConfig, size: int, count: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    return [torch.randn(size, generator=g, dtype=getattr(torch, cfg.dtype)).to(cfg.device)
            for _ in range(count)]


def _cycle(levels, *, omega: float, n_pre: int, n_post: int, inner=None, outer=None):
    """One V-cycle as a function of the residual, its settings bound now, not at call time.

    With `inner` and `outer` dtypes the cycle runs at `inner` on a residual held at `outer`.
    """
    def apply(r):
        if inner is None:
            return v_cycle(levels, r, omega=omega, n_pre=n_pre, n_post=n_post)
        return v_cycle(levels, r.to(inner), omega=omega, n_pre=n_pre, n_post=n_post).to(outer)
    return apply


def _pcg_kw(cfg: DictConfig, max_iter: int) -> dict:
    v = cfg.vcycle
    return dict(rtol=v.rtol, max_iter=max_iter, omega=v.omega, n_pre=v.n_pre, n_post=v.n_post)


def vcycle_identity(cfg: DictConfig):
    """The graph cycle against reference.py on the same hierarchy, over a smoother grid."""
    e, rows = cfg.experiment, []
    for case in e.cases:
        for n in _sizes(cfg):
            levels = _levels(cfg, _matrix(cfg, case, n))
            b = _vectors(cfg, levels[0].A.shape[0], 1)[0]
            for omega in e.omegas:
                for n_pre, n_post in e.sweeps:
                    got = v_cycle(levels, b, omega=omega, n_pre=n_pre, n_post=n_post)
                    want = reference_vcycle(levels, b.cpu().numpy(), omega=omega,
                                            n_pre=n_pre, n_post=n_post)
                    rows.append(dict(case=case, n=n, levels=len(levels), omega=omega,
                                     n_pre=n_pre, n_post=n_post,
                                     rel_diff=relative_difference(got.cpu().numpy(), want)))
    diffs = [r["rel_diff"] for r in rows]
    headline = dict(rel_diff_max=max(diffs), rel_diff_min=min(diffs))
    return rows, headline, dict(roundoff_everywhere=max(diffs) <= e.roundoff)


def vcycle_spd(cfg: DictConfig):
    """R = P^T, symmetry of the cycle at equal and unequal counts, and positive Rayleigh."""
    e, rows = cfg.experiment, []
    for case in e.cases:
        for n in _sizes(cfg):
            levels = _levels(cfg, _matrix(cfg, case, n))
            vecs = _vectors(cfg, levels[0].A.shape[0], e.n_vectors)
            defect = transpose_defect(levels)
            for n_pre, n_post in e.sweeps:
                B = _cycle(levels, omega=cfg.vcycle.omega, n_pre=n_pre, n_post=n_post)
                rows.append(dict(case=case, n=n, levels=len(levels), n_pre=n_pre, n_post=n_post,
                                 transpose_defect=defect,
                                 symmetry_defect=symmetry_defect(B, vecs[0], vecs[1]),
                                 rayleigh_min=min_rayleigh(B, vecs)))
    equal = [r["symmetry_defect"] for r in rows if r["n_pre"] == r["n_post"]]
    unequal = [r["symmetry_defect"] for r in rows if r["n_pre"] != r["n_post"]]
    headline = dict(symmetry_equal_max=max(equal), symmetry_unequal_min=min(unequal),
                    symmetry_unequal_max=max(unequal),
                    rayleigh_min=min(r["rayleigh_min"] for r in rows))
    checks = dict(transpose_exact=all(r["transpose_defect"] == 0.0 for r in rows),
                  symmetric_at_equal_counts=max(equal) <= e.roundoff,
                  unsymmetric_at_unequal_counts=min(unequal) >= e.asymmetry_floor,
                  positive_rayleigh=headline["rayleigh_min"] > 0.0)
    return rows, headline, checks


def hierarchy_rebuild(cfg: DictConfig):
    """Two builds per seed, and every seed against the build from cfg.seed."""
    e, rows = cfg.experiment, []
    for case in e.cases:
        for n in _sizes(cfg):
            A = _matrix(cfg, case, n)
            reference = _levels(cfg, A)
            b = _vectors(cfg, A.shape[0], 1)[0]
            x_ref = pcg(reference, b, **_pcg_kw(cfg, e.pcg_max_iter)).x.cpu().numpy()
            for seed in e.seeds:
                first, second = _levels(cfg, A, seed=seed), _levels(cfg, A, seed=seed)
                res = pcg(first, b, **_pcg_kw(cfg, e.pcg_max_iter))
                rows.append(dict(case=case, n=n, seed=seed, levels=len(first),
                                 same_seed_bitwise=bitwise_equal(first, second),
                                 same_seed_drift=coarse_drift(first, second),
                                 drift=coarse_drift(first, reference),
                                 iterations=res.iterations, converged=res.converged,
                                 solution_diff=relative_difference(res.x.cpu().numpy(), x_ref)))
    others = [r for r in rows if r["seed"] != cfg.seed]
    headline = dict(drift_max=max(r["drift"] for r in others),
                    drift_min=min(r["drift"] for r in others),
                    solution_diff_max=max(r["solution_diff"] for r in others))
    checks = dict(same_seed_bitwise=all(r["same_seed_bitwise"] and r["same_seed_drift"] == 0.0
                                        for r in rows),
                  other_seeds_drift=all(r["drift"] > 0.0 for r in others),
                  all_converged=all(r["converged"] for r in rows))
    return rows, headline, checks


def stationary_vs_pcg(cfg: DictConfig):
    """A V-cycle built on each operator, as a stationary solver and inside CG."""
    e, v, rows = cfg.experiment, cfg.vcycle, []
    for case in e.cases:
        for n in _sizes(cfg):
            levels = _levels(cfg, _matrix(cfg, case, n))
            A, B = levels[0].A, _cycle(levels, omega=v.omega, n_pre=v.n_pre, n_post=v.n_post)
            b = _vectors(cfg, A.shape[0], 1)[0]
            ritz = ritz_values(A, B, b, steps=e.lanczos_steps)
            lo, hi = float(ritz[0]), float(ritz[-1])
            krylov = pcg(levels, b, **_pcg_kw(cfg, e.pcg_max_iter))
            steps = [(w, "fixed") for w in e.omegas]
            if e.optimal_omega:
                steps.append((2.0 / (lo + hi), "optimal"))
            for omega, rule in steps:
                s = stationary(A, B, b, omega=omega, rtol=v.rtol, max_iter=e.stationary_max_iter,
                               divergence_ratio=e.divergence_ratio)
                rows.append(dict(case=case, n=n, omega=omega, omega_rule=rule, ritz_min=lo,
                                 ritz_max=hi, predicted_radius=stationary_radius(lo, hi, omega),
                                 stationary_iterations=s.iterations,
                                 stationary_converged=s.converged,
                                 stationary_diverged=s.diverged,
                                 stationary_factor=convergence_factor(s.residuals,
                                                                      window=e.factor_window),
                                 pcg_iterations=krylov.iterations,
                                 pcg_converged=krylov.converged))
    headline = dict(ratio_min=min(r["stationary_iterations"] / r["pcg_iterations"] for r in rows),
                    ratio_max=max(r["stationary_iterations"] / r["pcg_iterations"] for r in rows))
    checks = dict(pcg_at_most_stationary=all(r["pcg_iterations"] <= r["stationary_iterations"]
                                             for r in rows),
                  pcg_converged=all(r["pcg_converged"] for r in rows),
                  stationary_converged=all(r["stationary_converged"] for r in rows))
    return rows, headline, checks


def stationary_mismatch(cfg: DictConfig):
    """A V-cycle built on the uniform operator, applied to the high-contrast ones."""
    e, v, rows = cfg.experiment, cfg.vcycle, []
    for n in _sizes(cfg):
        levels = _levels(cfg, _matrix(cfg, e.source, n))
        B = _cycle(levels, omega=v.omega, n_pre=v.n_pre, n_post=v.n_post)
        for target in e.targets:
            S = MessagePassingOperator(_matrix(cfg, target, n), dtype=getattr(torch, cfg.dtype),
                                       use_sparse_mm=v.use_sparse_mm).to(cfg.device)
            b = _vectors(cfg, S.shape[0], 1)[0]
            ritz = ritz_values(S, B, b, steps=e.lanczos_steps)
            lo, hi = float(ritz[0]), float(ritz[-1])
            krylov = pcg(levels, b, **_pcg_kw(cfg, e.pcg_max_iter), operator=S, precondition=B)
            steps = [(w, "fixed", e.stationary_max_iter) for w in e.omegas]
            steps.append((e.below_limit_fraction * 2.0 / hi, "below_limit", e.below_limit_max_iter))
            for omega, rule, cap in steps:
                s = stationary(S, B, b, omega=omega, rtol=v.rtol, max_iter=cap,
                               divergence_ratio=e.divergence_ratio)
                rows.append(dict(source=e.source, target=target, n=n, omega=omega,
                                 omega_rule=rule, ritz_min=lo, ritz_max=hi, omega_limit=2.0 / hi,
                                 predicted_radius=stationary_radius(lo, hi, omega),
                                 stationary_iterations=s.iterations,
                                 stationary_converged=s.converged,
                                 stationary_diverged=s.diverged,
                                 stationary_residual=s.residuals[-1],
                                 pcg_iterations=krylov.iterations,
                                 pcg_converged=krylov.converged))
    fixed = [r for r in rows if r["omega_rule"] == "fixed"]
    below = [r for r in rows if r["omega_rule"] == "below_limit"]
    headline = dict(omega_limit_max=max(r["omega_limit"] for r in rows),
                    pcg_iterations_max=max(r["pcg_iterations"] for r in rows))
    checks = dict(diverges_at_every_fixed_omega=all(r["stationary_diverged"] for r in fixed),
                  no_divergence_below_limit=not any(r["stationary_diverged"] for r in below),
                  pcg_converged=all(r["pcg_converged"] for r in rows))
    return rows, headline, checks


def mixed_precision(cfg: DictConfig):
    """fp64 CG with the fp64 cycle, with the same hierarchy cast to fp32, and all in fp32."""
    e, v, rows = cfg.experiment, cfg.vcycle, []
    outer, inner = getattr(torch, cfg.dtype), getattr(torch, e.inner_dtype)
    for case in e.cases:
        for n in _sizes(cfg):
            full = _levels(cfg, _matrix(cfg, case, n))
            low = cast_levels(full, inner)
            b = _vectors(cfg, full[0].A.shape[0], 1)[0]
            kw = _pcg_kw(cfg, v.max_iter)
            ref = pcg(full, b, **kw)
            mixed = pcg(full, b, **kw, precondition=_cycle(low, omega=v.omega, n_pre=v.n_pre,
                                                           n_post=v.n_post, inner=inner,
                                                           outer=outer))
            single = pcg(low, b.to(inner), **kw)
            rows.append(dict(case=case, n=n, fp64_iterations=ref.iterations,
                             fp64_converged=ref.converged, mixed_iterations=mixed.iterations,
                             mixed_converged=mixed.converged,
                             mixed_extra=mixed.iterations - ref.iterations,
                             mixed_solution_diff=relative_difference(mixed.x.cpu().numpy(),
                                                                     ref.x.cpu().numpy()),
                             fp32_iterations=single.iterations, fp32_converged=single.converged,
                             fp32_residual=single.residuals[-1]))
    extra = [r["mixed_extra"] for r in rows]
    lo, hi = e.extra_range
    headline = dict(mixed_extra_min=min(extra), mixed_extra_max=max(extra),
                    fp32_residual_min=min(r["fp32_residual"] for r in rows))
    checks = dict(fp64_and_mixed_converged=all(r["fp64_converged"] and r["mixed_converged"]
                                               for r in rows),
                  mixed_extra_in_range=lo <= min(extra) and max(extra) <= hi,
                  fp32_misses_rtol=not any(r["fp32_converged"] for r in rows))
    return rows, headline, checks


EXPERIMENTS = dict(vcycle_identity=vcycle_identity, vcycle_spd=vcycle_spd,
                   hierarchy_rebuild=hierarchy_rebuild, stationary_vs_pcg=stationary_vs_pcg,
                   stationary_mismatch=stationary_mismatch, mixed_precision=mixed_precision)


def run(cfg: DictConfig) -> dict:
    """Run one experiment and return its summary. Reads cfg only; writes no file."""
    name = cfg.experiment.name
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; available: {sorted(EXPERIMENTS)}")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(cfg.deterministic)
    t0 = time.perf_counter()
    try:
        rows, headline, checks = EXPERIMENTS[name](cfg)
    finally:
        torch.use_deterministic_algorithms(previous)
    wall = time.perf_counter() - t0
    c = OmegaConf.to_container(cfg, resolve=True)
    config = dict(experiment={k: val for k, val in c["experiment"].items() if k != "regression"},
                  vcycle=c["vcycle"], cases=c["cases"], seed=c["seed"], dtype=c["dtype"],
                  deterministic=c["deterministic"])
    return dict(experiment=name, profile=cfg.profile, config=config, rows=rows,
                headline=headline, checks=checks,
                environment=summary.environment(cfg.device, cfg.dtype),
                timing=dict(wall_s=wall))


@hydra.main(version_base=None, config_path="conf", config_name="experiments")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import to_absolute_path

    if cfg.mode not in ("run", "check", "refresh"):
        raise ValueError(f"mode must be run, check or refresh, not {cfg.mode!r}")
    result = run(cfg)
    name, env = cfg.experiment.name, result["environment"]
    out = Path(HydraConfig.get().runtime.output_dir) / f"{name}.summary.json"
    summary.write(result, out)
    print(f"experiment={name}  profile={cfg.profile}  device={env['device']} "
          f"({env['device_name']})  rows={len(result['rows'])}  "
          f"wall {result['timing']['wall_s']:.1f}s")
    for k, val in sorted(result["headline"].items()):
        print(f"  {k} = {val}")
    for k, ok in sorted(result["checks"].items()):
        print(f"  claim {k}: {'holds' if ok else 'DOES NOT HOLD'}")
    print(f"wrote {out}")

    expected = Path(to_absolute_path(cfg.expected_dir)) / f"{name}.{cfg.profile}.json"
    if cfg.mode == "refresh":
        summary.write(summary.expected_view(result), expected)
        print(f"refreshed {expected}")
    elif cfg.mode == "check":
        if not expected.is_file():
            raise FileNotFoundError(f"no expectation at {expected}; set expected_dir, or "
                                    "run from the repository root")
        rules = OmegaConf.to_container(cfg.experiment.regression, resolve=True)
        problems = summary.compare(result, json.loads(expected.read_text()), rules)
        for p in problems:
            print(f"  MISMATCH {p}")
        print(f"check against {expected.name}: {len(problems)} mismatch(es)")
        if problems:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
