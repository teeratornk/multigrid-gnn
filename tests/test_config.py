"""The configuration is the interface. A run must be describable by its config alone."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF = str(Path(__file__).resolve().parents[1] / "src" / "gnn4buoyancy" / "conf")


def _cfg(overrides=None):
    with initialize_config_dir(version_base=None, config_dir=CONF):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_composes():
    c = _cfg()
    assert c.case.name == "conjugate_cavity" and c.solver.name == "graph_mg"


@pytest.mark.parametrize("case", ["conjugate_cavity", "heat_sink", "engine_bore"])
@pytest.mark.parametrize("solver", ["graph_mg", "jacobi_ref"])
def test_every_group_combination_composes(case, solver):
    c = _cfg([f"case={case}", f"solver={solver}"])
    assert c.case.name == case and c.solver.name == solver


def test_overrides_reach_the_solver():
    c = _cfg(["case.params.n=17", "solver.omega=0.55", "solver.n_pre=3"])
    assert c.case.params.n == 17 and c.solver.omega == 0.55 and c.solver.n_pre == 3


def test_no_run_parameter_is_hardcoded_in_source():
    """Grep the source for numeric literals that should be configuration. A reviewer
    should be able to trust that the config is the whole story."""
    import re
    src = Path(__file__).resolve().parents[1] / "src" / "gnn4buoyancy"
    # the CLI's solver knobs, then the experiment knobs; a value may be a number or a list
    knobs = ("omega|rtol|max_iter|n_pre|n_post|max_levels"
             "|seed|seeds|sizes|omegas|sweeps|n_vectors|roundoff|asymmetry_floor"
             "|lanczos_steps|stationary_max_iter|pcg_max_iter|below_limit_fraction"
             "|below_limit_max_iter|divergence_ratio|factor_window|extra_range|tol|floor")
    offenders = []
    for f in src.glob("*.py"):
        if f.name == "cli.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#")[0]
            if re.search(rf"\b({knobs})\s*=\s*[\d.\[\-]", code):
                if "def " not in code:            # signature defaults are the contract
                    offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "run parameters set outside a signature default:\n" + "\n".join(offenders)


def test_resolved_config_is_serialisable():
    """It is written into every result file, so it must round-trip."""
    c = _cfg(["case=heat_sink"])
    d = OmegaConf.to_container(c, resolve=True)
    assert OmegaConf.create(d).case.params.contrast == 1000.0
