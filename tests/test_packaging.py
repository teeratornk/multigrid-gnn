"""The package has to build and to carry its configuration.

`pip install -e` does not exercise the wheel builder, so a packaging fault is invisible
to every other test here and to the whole README workflow. This suite found the wheel
uncompilable: `packages = ["src/gnn4buoyancy"]` already takes conf/, and a force-include offered
the same files again, so hatchling aborted. A deposit that cannot be installed
non-editably is not a deposit.

Marked slow: it shells out to a real build.
"""
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAMES = ("vcycle_identity", "vcycle_spd", "hierarchy_rebuild", "stationary_vs_pcg",
                    "stationary_mismatch", "mixed_precision")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_wheel_builds_and_contains_the_config_groups(tmp_path):
    # `uv build` provisions the build backend into a temporary environment, which needs
    # an index unless it is already cached. Compute nodes here have no route to one, and
    # an unbounded call simply hangs: the first version of this test wedged a GPU job
    # until it was cancelled. Bound it, and skip rather than fail when the environment
    # cannot provision a backend, because that is a property of the machine and not of
    # the package.
    try:
        r = subprocess.run(["uv", "build", "--out-dir", str(tmp_path)],
                           cwd=ROOT, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        pytest.skip("uv build timed out, most likely no route to a package index")
    if r.returncode != 0 and ("Network" in r.stderr or "offline" in r.stderr
                              or "Failed to fetch" in r.stderr):
        pytest.skip(f"uv build cannot reach an index here: {r.stderr.strip()[:200]}")
    assert r.returncode == 0, f"uv build failed:\n{r.stdout}\n{r.stderr}"

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    names = zipfile.ZipFile(wheels[0]).namelist()

    # The console script resolves Hydra's config_path against the installed module, so a
    # wheel without conf/ produces a `gnn4buoyancy` that works in a source checkout and fails
    # everywhere else. Check the group files, not just the directory.
    for want in ("gnn4buoyancy/conf/config.yaml",
                 "gnn4buoyancy/conf/case/conjugate_cavity.yaml",
                 "gnn4buoyancy/conf/case/heat_sink.yaml",
                 "gnn4buoyancy/conf/case/engine_bore.yaml",
                 "gnn4buoyancy/conf/solver/graph_mg.yaml",
                 "gnn4buoyancy/conf/solver/jacobi_ref.yaml",
                 "gnn4buoyancy/conf/experiments.yaml",
                 *(f"gnn4buoyancy/conf/experiment/{e}.yaml" for e in EXPERIMENT_NAMES),
                 *(f"gnn4buoyancy/{m}.py" for m in ("experiments", "reference", "stationary",
                                                    "diagnostics", "summary"))):
        assert want in names, f"{want} missing from the wheel:\n{sorted(names)}"

    # the second console script, which a wheel without its entry point would silently lack
    ep = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
    scripts = zipfile.ZipFile(wheels[0]).read(ep).decode()
    assert "gnn4buoyancy-experiment = gnn4buoyancy.experiments:main" in scripts, scripts

    # the expectations are data a check compares against, so the sdist must carry them
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert sdists, "no sdist produced"
    members = tarfile.open(sdists[0]).getnames()
    for e in EXPERIMENT_NAMES:
        for profile in ("quick", "paper"):
            want = f"experiments/expected/{e}.{profile}.json"
            assert any(m.endswith(want) for m in members), f"{want} missing from the sdist"


def test_declared_python_range_includes_the_running_interpreter():
    """A range that excludes the interpreter under test would mean the suite is
    certifying something the metadata forbids installing."""
    # read with a regex: tomllib is 3.11+, so this test used to fail on the 3.10 it certifies
    import re
    spec = re.search(r'^requires-python = "([^"]+)"$',
                     (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    lo, hi = spec.split(",")
    assert lo.strip() == ">=3.10" and hi.strip() == "<3.13", spec
    assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version


def test_metadata_declares_the_experiment_app_and_its_data():
    """The fast half of the build test above, which skips wherever no index is reachable.

    Read as text with regexes, not with tomllib, which Python 3.10 does not have.
    """
    import re
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^gnn4buoyancy-experiment = "gnn4buoyancy\.experiments:main"$', text, re.M)
    assert re.search(r'^include = \[[^\]]*"experiments"', text, re.M), "sdist lacks experiments"
    assert re.search(r'^\s*"gpu: ', text, re.M), "no gpu marker registered"
    assert sorted(p.stem for p in (ROOT / "src" / "gnn4buoyancy" / "conf" / "experiment")
                  .glob("*.yaml")) == sorted(EXPERIMENT_NAMES)
