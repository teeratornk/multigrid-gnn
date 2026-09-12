"""End-to-end: the construction has to solve the system, and to a stated tolerance."""
import numpy as np
import pytest
import torch

from gnn4buoyancy import build_case, build_hierarchy, pcg


def test_converges_and_residual_is_honest(cavity, rng):
    levels = build_hierarchy(cavity)
    b = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    r = pcg(levels, b, rtol=1e-10, max_iter=200)
    assert r.converged, f"did not converge in {r.iterations}"
    true_rel = (torch.linalg.vector_norm(b - levels[0].A(r.x))
                / torch.linalg.vector_norm(b)).item()
    # the reported residual must be the real one, not a recurrence that has drifted
    assert true_rel <= 1e-9, true_rel
    assert abs(true_rel - r.residuals[-1]) / max(true_rel, 1e-30) < 1e-3


def test_solution_solves_the_system(cavity, rng):
    import scipy.sparse.linalg as spla
    levels = build_hierarchy(cavity)
    b = rng.standard_normal(cavity.shape[0])
    got = pcg(levels, torch.as_tensor(b), rtol=1e-12, max_iter=400).x.numpy()
    want = spla.spsolve(cavity.tocsc(), b)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-8


def test_multigrid_beats_single_level(rng):
    """The hierarchy is the point. Without it the same CG needs far more iterations."""
    A = build_case("conjugate_cavity", n=48)
    b = torch.as_tensor(rng.standard_normal(A.shape[0]))
    mg = pcg(build_hierarchy(A, max_levels=10), b, rtol=1e-8, max_iter=500)
    one = pcg(build_hierarchy(A, max_levels=1), b, rtol=1e-8, max_iter=500)
    assert mg.converged and one.converged
    assert mg.iterations < one.iterations / 3, (mg.iterations, one.iterations)


def test_iteration_count_is_mesh_robust(rng):
    """Near mesh independence is what a multigrid preconditioner is for. Allow
    growth but not the O(n) blow-up a single-level method shows."""
    counts = []
    for n in (32, 64):
        A = build_case("conjugate_cavity", n=n)
        b = torch.as_tensor(rng.standard_normal(A.shape[0]))
        counts.append(pcg(build_hierarchy(A), b, rtol=1e-8, max_iter=300).iterations)
    assert counts[1] <= 2 * counts[0], counts


def test_conjugate_interface_still_converges(sink, rng):
    levels = build_hierarchy(sink)
    b = torch.as_tensor(rng.standard_normal(sink.shape[0]))
    r = pcg(levels, b, rtol=1e-8, max_iter=500)
    assert r.converged, f"conjugate case stalled at {r.residuals[-1]:.2e}"


def test_curved_interface_still_converges(bore, rng):
    """The engine section staircases its solid-fluid boundary, which is harder for
    smoothed aggregation than the fin array's straight interfaces."""
    levels = build_hierarchy(bore)
    b = torch.as_tensor(rng.standard_normal(bore.shape[0]))
    r = pcg(levels, b, rtol=1e-8, max_iter=500)
    assert r.converged, f"engine section stalled at {r.residuals[-1]:.2e}"


def test_zero_rhs_gives_zero(cavity):
    levels = build_hierarchy(cavity)
    r = pcg(levels, torch.zeros(cavity.shape[0]))
    assert r.iterations == 0 and torch.allclose(r.x, torch.zeros_like(r.x))


def test_residuals_decrease_monotonically(cavity, rng):
    levels = build_hierarchy(cavity)
    b = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    h = pcg(levels, b, rtol=1e-10, max_iter=200).residuals
    assert all(b_ <= 1.5 * a for a, b_ in zip(h, h[1:])), "residual history is erratic"


def test_unsymmetric_smoothing_is_refused(cavity, rng):
    """CG is only valid for a symmetric preconditioner, and a V-cycle is symmetric
    only when the pre- and post-smoothing counts match. n_pre and n_post are separate
    config keys, so `solver.n_pre=3` alone is reachable from the command line and used
    to return a converged-looking answer whose theory does not hold."""
    import pytest
    levels = build_hierarchy(cavity)
    b = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    with pytest.raises(ValueError, match="unsymmetric"):
        pcg(levels, b, n_pre=3, n_post=1)
    # the escape hatch still works, for anyone deliberately studying the effect
    assert pcg(levels, b, n_pre=3, n_post=1, allow_unsymmetric=True).converged


def test_vcycle_symmetry_is_what_the_guard_protects(cavity, rng):
    """Measure the property the guard exists for, so the number is on record."""
    from gnn4buoyancy import v_cycle
    levels = build_hierarchy(cavity)
    x = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    y = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    def asym(n_pre, n_post):
        Bx = v_cycle(levels, x, n_pre=n_pre, n_post=n_post)
        By = v_cycle(levels, y, n_pre=n_pre, n_post=n_post)
        a, c = torch.dot(Bx, y).item(), torch.dot(x, By).item()
        return abs(a - c) / max(abs(a), 1e-30)
    assert asym(2, 2) < 1e-12, "equal counts must be symmetric"
    assert asym(3, 1) > 1e-6, "unequal counts must be visibly unsymmetric"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reported_residual_is_true_in_both_precisions(dtype, rng):
    """The residual is tracked by a recurrence, which drifts. Unchecked, float32 reported
    5e-9 "converged" on the shipped defaults while the true relative residual was 7.6e-5,
    a factor of 7899. Check the claim where it actually fails, not only in the fp64
    corner where the recurrence happens to stay exact."""
    import pytest as _pytest
    A = build_case("heat_sink", n=32, fins=4, contrast=1e3)
    levels = build_hierarchy(A, dtype=dtype)
    b = torch.as_tensor(rng.standard_normal(A.shape[0]), dtype=dtype)
    r = pcg(levels, b, rtol=1e-8, max_iter=400)
    true_rel = (torch.linalg.vector_norm(b - levels[0].A(r.x))
                / torch.linalg.vector_norm(b)).item()
    assert abs(true_rel - r.residuals[-1]) <= 1e-3 * max(true_rel, 1e-30), (
        f"reported {r.residuals[-1]:.3e} but true {true_rel:.3e}")
    if r.converged:
        assert true_rel <= 1e-8, f"claimed converged at a true residual of {true_rel:.3e}"


def test_mixed_dtype_is_refused(cavity):
    """An fp32 operator fed an fp64 vector used to return fp64 carrying fp32 accuracy on
    the gather path, while the sparse_mm path raised. The two backends are documented as
    equivalent, so they must agree that this is an error."""
    import pytest as _pytest
    levels = build_hierarchy(cavity, dtype=torch.float32)
    b64 = torch.zeros(cavity.shape[0], dtype=torch.float64)
    with _pytest.raises(TypeError, match="dtype"):
        levels[0].A(b64)
