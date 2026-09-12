"""The layer the experiments stand on, checked against references that do not share its code."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from gnn4buoyancy import __version__, build_case, build_hierarchy, pcg, v_cycle
from gnn4buoyancy.diagnostics import bitwise_equal, coarse_drift, ritz_values
from gnn4buoyancy.hierarchy import cast_levels
from gnn4buoyancy.layers import MessagePassingOperator
from gnn4buoyancy.reference import reference_vcycle
from gnn4buoyancy.stationary import stationary

ROOT = Path(__file__).resolve().parents[1]


def _seeded(A, seed=0, **kw):
    """Every build here is seeded, as in the experiments; see test_same_seed_*."""
    np.random.seed(seed)
    return build_hierarchy(A, **kw)


def _cycle(levels):
    return lambda r: v_cycle(levels, r, omega=0.7, n_pre=2, n_post=2)


# --- the stationary iteration ------------------------------------------------------------

@pytest.mark.parametrize("omega", [1.0, 0.6])
def test_stationary_matches_numpy_richardson(cavity, rng, omega):
    """Every iterate of the torch loop against x <- x + omega B(b - Ax) written in numpy,
    with B the scipy reference cycle, so the two share nothing but the hierarchy."""
    levels = _seeded(cavity)
    b = rng.standard_normal(cavity.shape[0])
    got = stationary(levels[0].A, _cycle(levels), torch.as_tensor(b), omega=omega,
                     rtol=1e-10, max_iter=60, divergence_ratio=1e8)
    x, hist = np.zeros_like(b), [1.0]
    for _ in range(got.iterations):
        x = x + omega * reference_vcycle(levels, b - cavity @ x, omega=0.7, n_pre=2, n_post=2)
        hist.append(np.linalg.norm(b - cavity @ x) / np.linalg.norm(b))
    assert got.converged and not got.diverged
    assert np.linalg.norm(got.x.numpy() - x) / np.linalg.norm(x) < 1e-12
    # absolute, not relative: near rtol the residual is itself of round-off size
    assert np.abs(np.array(got.residuals) - np.array(hist)).max() < 1e-12


def test_stationary_stops_at_the_first_iterate_past_the_divergence_ratio(rng):
    """A cycle built on the uniform operator, iterated on the contrast 1e3 one."""
    uniform = _seeded(build_case("conjugate_cavity", n=16))
    S = build_case("heat_sink", n=16, fins=2, contrast=1e3)
    b = rng.standard_normal(S.shape[0])
    ratio = 1e6
    got = stationary(MessagePassingOperator(S), _cycle(uniform), torch.as_tensor(b), omega=1.0,
                     rtol=1e-10, max_iter=200, divergence_ratio=ratio)
    x, first = np.zeros_like(b), None
    for k in range(1, 201):
        x = x + reference_vcycle(uniform, b - S @ x, omega=0.7, n_pre=2, n_post=2)
        if np.linalg.norm(b - S @ x) / np.linalg.norm(b) > ratio:
            first = k
            break
    assert got.diverged and not got.converged
    assert first is not None and got.iterations == first
    assert got.residuals[-1] > ratio and all(np.isfinite(got.residuals))


# --- Ritz values -------------------------------------------------------------------------

def _pair(kind):
    sink = build_case("heat_sink", n=16, fins=2, contrast=1e3)
    if kind == "cavity":
        A = build_case("conjugate_cavity", n=16)
        return A, _seeded(A)
    if kind == "sink":
        return sink, _seeded(sink)
    return sink, _seeded(build_case("conjugate_cavity", n=16))   # cycle from another operator


@pytest.mark.parametrize("kind", ["cavity", "sink", "mismatch"])
def test_ritz_values_reach_the_dense_eigenvalues(kind, rng):
    """Extreme Ritz values of BA against eigenvalues of the dense product, 256 unknowns."""
    A, levels = _pair(kind)
    N = A.shape[0]
    B = np.column_stack([_cycle(levels)(torch.as_tensor(e)).numpy() for e in np.eye(N)])
    lam = np.sort(np.linalg.eigvals(B @ A.toarray()).real)
    ritz = ritz_values(MessagePassingOperator(A), _cycle(levels),
                       torch.as_tensor(rng.standard_normal(N)), steps=N)
    top, bottom = abs(ritz[-1] - lam[-1]) / lam[-1], abs(ritz[0] - lam[0]) / lam[0]
    msg = f"{kind}: top {ritz[-1]:.12g} vs {lam[-1]:.12g} ({top:.2e}), " \
          f"bottom {ritz[0]:.12g} vs {lam[0]:.12g} ({bottom:.2e})"
    assert lam[0] > 0, msg
    # Ritz values lie inside the spectrum, up to round-off
    assert lam[0] * (1 - 1e-10) <= ritz.min() and ritz.max() <= lam[-1] * (1 + 1e-10), msg
    # Bounds are about 10x the measured errors. A cycle on its own operator puts the top of BA
    # in a dense cluster at 1, which Lanczos resolves only to ~3e-4 before CG converges. The
    # mismatch top is isolated, and it is the lambda_max that sets the stability limit, so it
    # must match to round-off. Measured, top then bottom: cavity 2.6e-4 and 8.2e-6, sink
    # 2.3e-4 and 2.0e-10, mismatch 2.6e-15 and 1.0e-5.
    top_bound, bottom_bound = {"cavity": (3e-3, 1e-4), "sink": (3e-3, 1e-8),
                               "mismatch": (1e-13, 2e-4)}[kind]
    assert top < top_bound and bottom < bottom_bound, msg


# --- rebuild reproducibility -------------------------------------------------------------

@pytest.mark.parametrize("name,kw", [("conjugate_cavity", dict(n=32)),
                                     ("heat_sink", dict(n=32, fins=4, contrast=1e3)),
                                     ("engine_bore", dict(n=32, contrast=1e3))])
def test_same_seed_builds_are_bitwise_equal(name, kw):
    A = build_case(name, **kw)
    first, second = _seeded(A, seed=3), _seeded(A, seed=3)
    assert bitwise_equal(first, second)
    assert coarse_drift(first, second) == 0.0


def test_another_seed_drifts_so_the_check_is_not_vacuous(cavity):
    first, other = _seeded(cavity, seed=0), _seeded(cavity, seed=1)
    assert not bitwise_equal(first, other)
    assert coarse_drift(first, other) > 0.0


def test_bitwise_equal_sees_one_changed_bit(cavity):
    levels, copy = _seeded(cavity), _seeded(cavity)
    w = copy[-1].A.weight
    w[0] = torch.nextafter(w[0], w[0] + 1)
    assert not bitwise_equal(levels, copy)
    assert coarse_drift(levels, copy) > 0.0


# --- cast_levels -------------------------------------------------------------------------

def test_cast_to_the_held_dtype_is_the_same_cycle(sink, rng):
    levels = _seeded(sink)
    b = torch.as_tensor(rng.standard_normal(sink.shape[0]))
    assert torch.equal(v_cycle(levels, b), v_cycle(cast_levels(levels, torch.float64), b))


def test_cast_of_a_seeded_build_is_the_seeded_fp32_build(sink):
    """So mixed_precision changes precision and nothing else."""
    cast = cast_levels(_seeded(sink), torch.float32)
    assert all(t.dtype == torch.float32 for lv in cast for t in (lv.A.weight, lv.diag_inv))
    assert bitwise_equal(cast, _seeded(sink, dtype=torch.float32))


# --- pcg default path --------------------------------------------------------------------

def _pcg_before(levels, b, *, rtol=1e-8, max_iter=200, omega=0.7, n_pre=2, n_post=2):
    """pcg.py as it was before `operator=` and `precondition=`, code unchanged, comments cut."""
    A = levels[0].A
    x = torch.zeros_like(b)
    r = b - A(x)
    b_norm = torch.linalg.vector_norm(b).item() or 1.0
    hist = [torch.linalg.vector_norm(r).item() / b_norm]
    if hist[0] <= rtol:
        return x, 0, hist, True
    z = v_cycle(levels, r, omega=omega, n_pre=n_pre, n_post=n_post)
    p, rz = z.clone(), torch.dot(r, z)
    for k in range(1, max_iter + 1):
        Ap = A(p)
        alpha = rz / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = torch.linalg.vector_norm(r).item() / b_norm
        hist.append(rel)
        if rel <= rtol:
            r = b - A(x)
            rel = torch.linalg.vector_norm(r).item() / b_norm
            hist[-1] = rel
            if rel <= rtol:
                return x, k, hist, True
            z = v_cycle(levels, r, omega=omega, n_pre=n_pre, n_post=n_post)
            p, rz = z.clone(), torch.dot(r, z)
            continue
        z = v_cycle(levels, r, omega=omega, n_pre=n_pre, n_post=n_post)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    rel = (torch.linalg.vector_norm(b - A(x)).item() / b_norm)
    hist[-1] = rel
    return x, max_iter, hist, rel <= rtol


@pytest.mark.parametrize("name,kw,dtype,rtol", [
    ("conjugate_cavity", dict(n=32), torch.float64, 1e-10),
    ("engine_bore", dict(n=32, contrast=1e3), torch.float64, 1e-8),
    # fp32 never meets 1e-8, so this one runs the true-residual restart branch to max_iter
    ("heat_sink", dict(n=32, fins=4, contrast=1e3), torch.float32, 1e-8),
])
def test_pcg_default_path_is_bitwise_unchanged(name, kw, dtype, rtol, rng):
    A = build_case(name, **kw)
    levels = _seeded(A, dtype=dtype)
    b = torch.as_tensor(rng.standard_normal(A.shape[0]), dtype=dtype)
    x, iterations, hist, converged = _pcg_before(levels, b, rtol=rtol, max_iter=300)
    if dtype == torch.float32:
        assert not converged, "the restart branch was not exercised"
    for res in (pcg(levels, b, rtol=rtol, max_iter=300),
                pcg(levels, b, rtol=rtol, max_iter=300, operator=levels[0].A,
                    precondition=_cycle(levels))):
        assert torch.equal(res.x, x)
        assert (res.iterations, res.residuals, res.converged) == (iterations, hist, converged)


# --- the reference cycle and the version -------------------------------------------------

def _reviewed_reference():
    """reference_vcycle from tests/test_vcycle.py, loaded by path so no import mode matters."""
    spec = importlib.util.spec_from_file_location("_reviewed_test_vcycle",
                                                  ROOT / "tests" / "test_vcycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference_vcycle


@pytest.mark.parametrize("omega,n_pre,n_post", [(0.7, 2, 2), (0.5, 1, 1), (0.9, 3, 1)])
def test_package_reference_is_the_reviewed_test_reference(bore, rng, omega, n_pre, n_post):
    levels = _seeded(bore)
    b = rng.standard_normal(bore.shape[0])
    want = _reviewed_reference()(levels, b, omega, n_pre, n_post)
    got = reference_vcycle(levels, b, omega=omega, n_pre=n_pre, n_post=n_post)
    assert got.tobytes() == want.tobytes()


def test_version_is_the_same_everywhere():
    # read with regexes, not tomllib, which Python 3.10 does not have
    def first(pattern, path):
        return re.search(pattern, (ROOT / path).read_text(), re.M).group(1)
    found = {"src/gnn4buoyancy/__init__.py": __version__,
             "pyproject.toml": first(r'^version = "([^"]+)"$', "pyproject.toml"),
             "CITATION.cff": first(r"^version:\s*(\S+)\s*$", "CITATION.cff")}
    lock = ROOT / "uv.lock"                    # not in the sdist, so only when present
    if lock.is_file():
        found["uv.lock"] = re.search(r'name = "gnn4buoyancy-mwe"\nversion = "([^"]+)"',
                                     lock.read_text()).group(1)
    assert len(set(found.values())) == 1, found


def test_licence_is_the_same_everywhere():
    """LICENSE, pyproject.toml and CITATION.cff name one licence, so no index shows another."""
    licence = (ROOT / "LICENSE").read_text()
    assert licence.splitlines()[0] == "MIT License"
    assert "Permission is hereby granted, free of charge" in licence
    found = {"pyproject.toml": re.search(r'^license = "([^"]+)"$',
                                         (ROOT / "pyproject.toml").read_text(), re.M).group(1),
             "CITATION.cff": re.search(r"^license:\s*(\S+)\s*$",
                                       (ROOT / "CITATION.cff").read_text(), re.M).group(1)}
    assert set(found.values()) == {"MIT"}, found
