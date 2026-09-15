"""A check that cannot fail proves nothing.

Each test here breaks the property one experiment claims, on a small CPU grid, and shows
that the claim then reads false. The same settings unbroken are run first, so the flip is
the mutation's doing and not the smaller grid's. The last tests corrupt expectation files
beyond each stated tolerance and show that the comparison, and check mode, catch it.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gnn4buoyancy import experiments, summary
from gnn4buoyancy.hierarchy import build_hierarchy
from gnn4buoyancy.pcg import pcg
from gnn4buoyancy.vcycle import v_cycle

ROOT = Path(__file__).resolve().parents[1]
CONF = str(ROOT / "src" / "gnn4buoyancy" / "conf")
EXPECTED = ROOT / "experiments" / "expected"
NAMES = sorted(experiments.EXPERIMENTS)
SMALL = "experiment.profiles.quick.sizes=[16]"


def _cfg(name, *overrides):
    with initialize_config_dir(version_base=None, config_dir=CONF):
        return compose(config_name="experiments",
                       overrides=[f"experiment={name}", "profile=quick", SMALL, *overrides])


def _checks(name, *overrides):
    return experiments.run(_cfg(name, *overrides))["checks"]


@pytest.mark.parametrize("name", NAMES)
def test_the_small_grid_holds_before_any_mutation(name):
    checks = _checks(name)
    assert all(checks.values()), checks


# --- vcycle_identity ---------------------------------------------------------------------

@pytest.mark.parametrize("mutation", ["skip_the_coarsest_level", "smoother_weight_1e-9"])
def test_identity_fails_when_the_graph_execution_changes(monkeypatch, mutation):
    def skipped(levels, b, **kw):
        return v_cycle(levels[:-1], b, **kw)

    def reweighted(levels, b, *, omega, **kw):
        return v_cycle(levels, b, omega=omega * (1 + 1e-9), **kw)

    monkeypatch.setattr(experiments, "v_cycle",
                        skipped if mutation == "skip_the_coarsest_level" else reweighted)
    assert _checks("vcycle_identity")["roundoff_everywhere"] is False


# --- vcycle_spd --------------------------------------------------------------------------

def test_spd_fails_when_one_entry_of_R_moves(monkeypatch):
    original = experiments._levels

    def perturbed(cfg, A, *, seed=None):
        levels = original(cfg, A, seed=seed)
        levels[0].R.weight[0] *= 1 + 1e-6
        return levels

    monkeypatch.setattr(experiments, "_levels", perturbed)
    checks = _checks("vcycle_spd")
    assert checks["transpose_exact"] is False
    assert checks["symmetric_at_equal_counts"] is False


def test_spd_fails_when_the_cycle_is_negated(monkeypatch):
    original = experiments._cycle

    def negated(levels, **kw):
        B = original(levels, **kw)
        return lambda r: -B(r)

    monkeypatch.setattr(experiments, "_cycle", negated)
    assert _checks("vcycle_spd")["positive_rayleigh"] is False


def test_spd_fails_when_unequal_counts_are_quietly_equalised(monkeypatch):
    original = experiments._cycle

    def equalised(levels, *, omega, n_pre, n_post, **kw):
        return original(levels, omega=omega, n_pre=n_pre, n_post=n_pre, **kw)

    monkeypatch.setattr(experiments, "_cycle", equalised)
    assert _checks("vcycle_spd")["unsymmetric_at_unequal_counts"] is False


# --- hierarchy_rebuild -------------------------------------------------------------------

def test_rebuild_fails_when_builds_are_not_seeded(monkeypatch):
    def unseeded(cfg, A, *, seed=None):
        np.random.seed(None)               # fresh OS entropy, as the CLI's own builds get
        return build_hierarchy(A, max_levels=cfg.vcycle.max_levels,
                               dtype=getattr(torch, cfg.dtype), device=cfg.device,
                               use_sparse_mm=cfg.vcycle.use_sparse_mm)

    monkeypatch.setattr(experiments, "_levels", unseeded)
    assert _checks("hierarchy_rebuild")["same_seed_bitwise"] is False


def test_rebuild_fails_when_the_seed_is_ignored(monkeypatch):
    original = experiments._levels
    monkeypatch.setattr(experiments, "_levels", lambda cfg, A, *, seed=None: original(cfg, A))
    checks = _checks("hierarchy_rebuild")
    assert checks["other_seeds_drift"] is False
    assert checks["same_seed_bitwise"] is True


def test_rebuild_fails_when_a_solve_is_cut_short():
    assert _checks("hierarchy_rebuild", "experiment.pcg_max_iter=2")["all_converged"] is False


# --- stationary_vs_pcg -------------------------------------------------------------------

def test_krylov_claim_fails_without_the_preconditioner(monkeypatch):
    monkeypatch.setattr(experiments, "pcg",
                        lambda levels, b, **kw: pcg(levels, b, **kw, precondition=lambda r: r))
    assert _checks("stationary_vs_pcg")["pcg_at_most_stationary"] is False


def test_stationary_claim_fails_when_cut_short():
    checks = _checks("stationary_vs_pcg", "experiment.stationary_max_iter=2")
    assert checks["stationary_converged"] is False


# --- stationary_mismatch -----------------------------------------------------------------

def test_divergence_claim_fails_when_the_cycle_matches_its_operator():
    checks = _checks("stationary_mismatch", "experiment.source=heat_sink",
                     "experiment.targets=[heat_sink]")
    assert checks["diverges_at_every_fixed_omega"] is False


def test_limit_claim_fails_above_the_stability_limit():
    checks = _checks("stationary_mismatch", "experiment.below_limit_fraction=1.5")
    assert checks["no_divergence_below_limit"] is False


# --- mixed_precision ---------------------------------------------------------------------

def test_fp32_claim_fails_when_the_inner_dtype_is_fp64():
    assert _checks("mixed_precision", "experiment.inner_dtype=float64")["fp32_misses_rtol"] is False


def test_mixed_claim_fails_when_the_fp32_cycle_is_weakened(monkeypatch):
    original = experiments.cast_levels

    def weakened(levels, dtype):
        low = original(levels, dtype)
        for lv in low:
            lv.diag_inv.mul_(0.1)          # a tenth of the smoother weight on every level
        return low

    monkeypatch.setattr(experiments, "cast_levels", weakened)
    assert _checks("mixed_precision")["mixed_extra_in_range"] is False


# --- expectations ------------------------------------------------------------------------

def _beyond(value, rule):
    """A value just outside what `rule` accepts around `value`."""
    kind, tol = rule["rule"], rule.get("tol", 0)
    if isinstance(value, bool):
        return not value
    if kind == "abs":
        return value + 2 * tol + 1
    if kind == "rel":
        return value * (1 + 2 * tol) if value else 1.0
    if kind == "decades":
        return max(abs(value), rule["floor"]) * 10 ** (2 * tol + 1)
    return value + 1


@pytest.mark.parametrize("profile", ["quick", "paper"])
@pytest.mark.parametrize("name", NAMES)
def test_every_rule_catches_a_change_beyond_its_tolerance(name, profile):
    want = json.loads((EXPECTED / f"{name}.{profile}.json").read_text())
    rules = OmegaConf.to_container(_cfg(name).experiment.regression, resolve=True)
    assert rules, f"{name} has no regression rules, so every field compares exactly"
    for field, rule in rules.items():
        bad = json.loads(json.dumps(want))
        where = bad["headline"] if field in bad["headline"] else next(
            r for r in bad["rows"] if r.get(field) is not None)
        where[field] = _beyond(where[field], rule)
        problems = summary.compare(want, bad, rules)
        assert len(problems) == 1 and field in problems[0], (field, problems)


@pytest.mark.parametrize("name", NAMES)
def test_an_unruled_field_must_match_exactly(name):
    want = json.loads((EXPECTED / f"{name}.quick.json").read_text())
    rules = OmegaConf.to_container(_cfg(name).experiment.regression, resolve=True)
    bad = json.loads(json.dumps(want))
    row = bad["rows"][0]
    field = next(k for k, v in sorted(row.items()) if k not in rules and isinstance(v, (int, str)))
    row[field] = _beyond(row[field], {"rule": "exact"}) if not isinstance(row[field], str) else "x"
    problems = summary.compare(want, bad, rules)
    assert len(problems) == 1 and field in problems[0], (field, problems)


def test_check_mode_exits_nonzero_on_a_corrupted_expectation(tmp_path):
    name = "vcycle_identity"
    bad = json.loads((EXPECTED / f"{name}.quick.json").read_text())
    bad["rows"][0]["levels"] += 1
    (tmp_path / "expected").mkdir()
    (tmp_path / "expected" / f"{name}.quick.json").write_text(summary.dumps(bad))
    r = subprocess.run([sys.executable, "-m", "gnn4buoyancy.experiments", f"experiment={name}",
                        "profile=quick", "mode=check", f"expected_dir={tmp_path / 'expected'}",
                        f"hydra.run.dir={tmp_path / 'run'}"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "MISMATCH" in r.stdout and "levels" in r.stdout, r.stdout


# --- the comparator itself ---------------------------------------------------------------

def _inside(value, rule):
    """A value half a tolerance away from `value`, which every rule must accept."""
    kind, tol = rule["rule"], rule.get("tol", 0)
    if kind == "abs":
        return value + 0.5 * tol
    if kind == "rel":
        return value * (1 + 0.5 * tol)
    if kind == "decades":
        return max(abs(value), rule["floor"]) * 10 ** (0.5 * tol)
    return value


def _nearly_outside(value, rule):
    """Three quarters of a tolerance away: accepted by the stated tolerance, rejected by one
    that was halved, so a comparator that quietly tightened is caught as well as one that
    loosened."""
    kind, tol = rule["rule"], rule.get("tol", 0)
    if kind == "abs":
        return value + 0.75 * tol
    if kind == "rel":
        return value * (1 + 0.75 * tol)
    if kind == "decades":
        return max(abs(value), rule["floor"]) * 10 ** (0.75 * tol)
    return value


def _outside(value, rule):
    """A value one and a half tolerances away, which every rule must reject."""
    kind, tol = rule["rule"], rule.get("tol", 0)
    if kind == "abs":
        return value + 1.5 * tol
    if kind == "rel":
        return value * (1 + 1.5 * tol)
    if kind == "decades":
        return max(abs(value), rule["floor"]) * 10 ** (1.5 * tol)
    return value + 1


def _holder(record, field):
    return record["headline"] if field in record["headline"] else next(
        r for r in record["rows"] if r.get(field) is not None)


@pytest.mark.parametrize("name", NAMES)
def test_every_tolerance_is_exactly_the_one_the_config_states(name):
    """Half and three quarters of a tolerance away pass and one and a half away fails, so a
    comparator whose tolerance drifted by a factor of two either way is caught, not only one
    that is far off."""
    want = json.loads((EXPECTED / f"{name}.quick.json").read_text())
    rules = OmegaConf.to_container(_cfg(name).experiment.regression, resolve=True)
    tested = 0
    for field, rule in rules.items():
        value = _holder(want, field)[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if rule["rule"] == "rel" and value == 0:
            continue
        for make, expect in ((_inside, 0), (_nearly_outside, 0), (_outside, 1)):
            bad = json.loads(json.dumps(want))
            _holder(bad, field)[field] = make(value, rule)
            problems = summary.compare(want, bad, rules)
            assert len(problems) == expect, (field, make.__name__, problems)
        tested += 1
    assert tested, name


def test_compare_reports_changed_checks_config_and_lost_rows():
    """A changed claim verdict, a changed configuration and a missing row are each a
    reported difference, not something the row-by-row field comparison walks past."""
    want = json.loads((EXPECTED / "vcycle_identity.quick.json").read_text())
    rules = OmegaConf.to_container(_cfg("vcycle_identity").experiment.regression, resolve=True)
    assert summary.compare(want, want, rules) == []
    for key in ("checks", "config"):
        bad = json.loads(json.dumps(want))
        bad[key] = dict(bad[key], review="changed")
        assert any(p.startswith(key) for p in summary.compare(want, bad, rules)), key
    bad = json.loads(json.dumps(want))
    bad["rows"] = bad["rows"][:-1]
    assert any(p.startswith("rows") for p in summary.compare(want, bad, rules))
