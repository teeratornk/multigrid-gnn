"""The experiment app: its configs, its claims on the quick profile, and its expectations."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gnn4buoyancy import experiments, summary

ROOT = Path(__file__).resolve().parents[1]
CONF = str(ROOT / "src" / "gnn4buoyancy" / "conf")
EXPECTED = ROOT / "experiments" / "expected"
NAMES = sorted(experiments.EXPERIMENTS)
PROFILES = ("quick", "paper")


def _cfg(name, profile, *overrides, config="experiments"):
    with initialize_config_dir(version_base=None, config_dir=CONF):
        return compose(config_name=config,
                       overrides=[f"experiment={name}", f"profile={profile}", *overrides])


def _rules(name):
    return OmegaConf.to_container(_cfg(name, "quick").experiment.regression, resolve=True)


def _expected(name, profile):
    return json.loads((EXPECTED / f"{name}.{profile}.json").read_text())


# --- configuration -----------------------------------------------------------------------

def test_every_experiment_has_one_config_file():
    assert sorted(p.stem for p in (Path(CONF) / "experiment").glob("*.yaml")) == NAMES


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", NAMES)
def test_configs_compose(name, profile):
    c = _cfg(name, profile)
    assert c.experiment.name == name and c.profile == profile
    assert c.device == c.experiment.profiles[profile].device
    assert sorted(c.cases) == ["conjugate_cavity", "engine_bore", "heat_sink"]
    OmegaConf.to_container(c, resolve=True)          # every interpolation resolves


def test_profiles_are_the_documented_ones():
    for name in NAMES:
        p = _cfg(name, "quick").experiment.profiles
        assert (list(p.quick.sizes), p.quick.device) == ([16, 32], "cpu"), name
        assert (list(p.paper.sizes), p.paper.device) == ([32, 64, 128], "cuda"), name


def test_experiments_use_the_cli_groups_unchanged():
    """solver@vcycle and case@cases.* are the CLI's own files, so an experiment runs the
    configuration a user runs, not a copy that can drift from it."""
    with initialize_config_dir(version_base=None, config_dir=CONF):
        cli = compose(config_name="config")
    c = _cfg(NAMES[0], "quick")
    assert c.vcycle == cli.solver
    for case in c.cases:
        with initialize_config_dir(version_base=None, config_dir=CONF):
            assert c.cases[case] == compose(config_name="config", overrides=[f"case={case}"]).case


@pytest.mark.parametrize("name", NAMES)
def test_regression_rules_name_real_fields(name):
    data = _expected(name, "quick")
    fields = set().union(*data["rows"]) | set(data["headline"])
    for field, rule in _rules(name).items():
        assert field in fields, f"{name}: rule for {field!r}, which no row or headline has"
        assert rule["rule"] in ("exact", "abs", "rel", "decades"), rule


# --- the quick profile, run once per module ----------------------------------------------

@pytest.fixture(scope="module")
def quick():
    return {name: experiments.run(_cfg(name, "quick")) for name in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_quick_claims_hold(quick, name):
    failed = sorted(k for k, ok in quick[name]["checks"].items() if not ok)
    assert not failed, f"{name}: claims that do not hold: {failed}"


@pytest.mark.parametrize("name", NAMES)
def test_quick_matches_expected(quick, name):
    problems = summary.compare(quick[name], _expected(name, "quick"), _rules(name))
    assert not problems, "\n".join(problems)


def test_summary_records_the_environment_and_check_ignores_it(quick):
    s = quick["vcycle_identity"]
    for key in ("package_version", "python", "torch", "cuda", "device", "device_name", "dtype"):
        assert key in s["environment"], key
    moved = {**s, "environment": {**s["environment"], "torch": "0"}, "timing": {"wall_s": -1.0}}
    assert summary.compare(moved, _expected("vcycle_identity", "quick"), _rules("vcycle_identity")) == []


def test_check_catches_a_changed_row(quick):
    """Otherwise a passing comparison would mean nothing."""
    s = quick["stationary_vs_pcg"]
    rows = [dict(r) for r in s["rows"]]
    rows[0]["pcg_iterations"] += 1000     # far outside any tolerance in the config
    problems = summary.compare({**s, "rows": rows}, _expected("stationary_vs_pcg", "quick"),
                               _rules("stationary_vs_pcg"))
    assert len(problems) == 1 and "pcg_iterations" in problems[0], problems


# --- committed expectations --------------------------------------------------------------

@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", NAMES)
def test_expected_files_are_well_formed(name, profile):
    text = (EXPECTED / f"{name}.{profile}.json").read_text()
    data = json.loads(text)
    assert text == summary.dumps(data), "not canonical: sorted keys, two-space indent, no NaN"
    assert sorted(data) == sorted(summary.MEASURED_KEYS), "environment or timing leaked in"
    assert (data["experiment"], data["profile"]) == (name, profile)
    assert data["rows"] and all(isinstance(v, bool) for v in data["checks"].values())
    assert all(data["checks"].values()), "an expectation records a claim that does not hold"
    for word in ("/home", "/tmp", "outputs/", "result.json", "slurm", "wall_s", "device_name",
                 "hostname", "DOE"):
        assert word not in text, word


# --- the Hydra app itself, and the GPU profile -------------------------------------------

def test_app_writes_a_summary_and_a_refreshed_expectation(tmp_path):
    name = "vcycle_identity"
    r = subprocess.run([sys.executable, "-m", "gnn4buoyancy.experiments", f"experiment={name}",
                        "profile=quick", "mode=refresh", f"expected_dir={tmp_path / 'expected'}",
                        f"hydra.run.dir={tmp_path / 'run'}"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    run = json.loads((tmp_path / "run" / f"{name}.summary.json").read_text())
    refreshed = json.loads((tmp_path / "expected" / f"{name}.quick.json").read_text())
    assert "environment" in run and "timing" in run and "environment" not in refreshed
    assert summary.compare(run, refreshed, _rules(name)) == []
    assert summary.compare(run, _expected(name, "quick"), _rules(name)) == []


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("name", NAMES)
def test_paper_on_gpu_matches_expected(name):
    got = experiments.run(_cfg(name, "paper"))
    assert got["environment"]["device"] == "cuda"
    assert all(got["checks"].values()), got["checks"]
    problems = summary.compare(got, _expected(name, "paper"), _rules(name))
    assert not problems, "\n".join(problems)
