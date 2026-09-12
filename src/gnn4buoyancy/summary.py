"""Run summaries, and the comparison against committed expectations.

A summary has two halves. The measured half (experiment, profile, config, rows, headline,
checks) is what `experiments/expected/<name>.<profile>.json` holds. The environment half
(package and library versions, device, dtype, wall time) says where a run happened. A
reader needs it and a regression check must ignore it, so it is stripped before anything
is compared or committed. No field holds a host name or a path.
"""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import torch

ENVIRONMENT_KEYS = ("environment", "timing")
MEASURED_KEYS = ("checks", "config", "experiment", "headline", "profile", "rows")


def environment(device: str, dtype: str) -> dict:
    import hydra
    import numpy
    import pyamg
    import scipy

    from . import __version__
    dev = torch.device(device)
    name = (torch.cuda.get_device_name(dev) if dev.type == "cuda"
            else platform.processor() or platform.machine())
    return dict(package_version=__version__, python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda, numpy=numpy.__version__,
                scipy=scipy.__version__, pyamg=pyamg.__version__, hydra=hydra.__version__,
                device=device, device_name=name, dtype=dtype)


def dumps(summary: dict) -> str:
    """Sorted keys, and no NaN or infinity, which strict JSON readers reject."""
    return json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(summary))


def expected_view(summary: dict) -> dict:
    """The measured half, normalised through JSON so it compares like a loaded file."""
    return json.loads(dumps({k: v for k, v in summary.items() if k not in ENVIRONMENT_KEYS}))


def _agrees(got, want, rule: dict | None) -> bool:
    """exact | abs (|got - want| <= tol) | rel (<= tol |want|) | decades.

    `decades` is for magnitudes spanning round-off to order one: both values are raised to
    `floor`, then may differ by `tol` powers of ten. Below the floor digits are noise.
    """
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (got, want))
    if rule is None or rule["rule"] == "exact" or not numeric:
        return got == want
    if rule["rule"] == "abs":
        return abs(got - want) <= rule["tol"]
    if rule["rule"] == "rel":
        return abs(got - want) <= rule["tol"] * abs(want)
    if rule["rule"] == "decades":
        lo = rule["floor"]
        return abs(math.log10(max(abs(got), lo)) - math.log10(max(abs(want), lo))) <= rule["tol"]
    raise ValueError(f"unknown regression rule {rule['rule']!r}")


def _fields(label: str, got: dict, want: dict, rules: dict) -> list[str]:
    if sorted(got) != sorted(want):
        return [f"{label}: fields {sorted(got)} != expected {sorted(want)}"]
    return [f"{label} {k}: expected {want[k]!r}, got {got[k]!r} (rule {rules.get(k)})"
            for k in sorted(want) if not _agrees(got[k], want[k], rules.get(k))]


def compare(summary: dict, expected: dict, rules: dict) -> list[str]:
    """Every disagreement between a run and its expectation, as readable lines.

    Rows and headline fields use the rule named for that field in the experiment's
    `regression:` block; a field with no rule, and everything else, must match exactly.
    """
    got = expected_view(summary)
    if sorted(got) != sorted(expected):
        return [f"top-level keys {sorted(got)} != expected {sorted(expected)}"]
    problems = [f"{k}: expected {expected[k]!r}, got {got[k]!r}"
                for k in ("checks", "config", "experiment", "profile") if got[k] != expected[k]]
    problems += _fields("headline", got["headline"], expected["headline"], rules)
    if len(got["rows"]) != len(expected["rows"]):
        return problems + [f"rows: {len(got['rows'])} != expected {len(expected['rows'])}"]
    for i, (g, w) in enumerate(zip(got["rows"], expected["rows"])):
        problems += _fields(f"rows[{i}]", g, w, rules)
    return problems
