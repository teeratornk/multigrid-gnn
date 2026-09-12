"""Does the graph V-cycle execute the algebra it represents?

This is the paper's method-preservation claim at the operator level. The reference
is a numpy implementation of the SAME cycle on the SAME hierarchy: weighted Jacobi,
same omega and sweep counts, coarsest level smoothed rather than solved. PyAMG's own
preconditioner is a different cycle (Gauss-Seidel, direct coarse solve) and would be
the wrong reference.
"""
import numpy as np
import pytest
import torch

from gnn4buoyancy import build_hierarchy, v_cycle


def reference_vcycle(levels, b, omega, n_pre, n_post):
    """The same cycle, in numpy, on the hierarchy the graph object holds."""
    A = [l.A.to_scipy() for l in levels]
    dinv = [l.diag_inv.numpy() for l in levels]
    R = [l.R.to_scipy() if l.R is not None else None for l in levels]
    P = [l.P.to_scipy() if l.P is not None else None for l in levels]

    def smooth(k, x, rhs, n):
        for _ in range(n):
            x = x + omega * dinv[k] * (rhs - A[k] @ x)
        return x

    def rec(k, rhs):
        x = smooth(k, np.zeros_like(rhs), rhs, n_pre)
        if k == len(levels) - 1:
            return x
        x = x + P[k] @ rec(k + 1, R[k] @ (rhs - A[k] @ x))
        return smooth(k, x, rhs, n_post)

    return rec(0, b)


@pytest.mark.parametrize("omega,n_pre,n_post", [(0.7, 2, 2), (0.5, 1, 1), (0.9, 3, 1)])
def test_graph_vcycle_matches_reference(cavity, rng, omega, n_pre, n_post):
    """Must hold for any smoother setting: it is a property of the execution,
    not of one tuned configuration."""
    levels = build_hierarchy(cavity)
    b = rng.standard_normal(cavity.shape[0])
    want = reference_vcycle(levels, b, omega, n_pre, n_post)
    got = v_cycle(levels, torch.as_tensor(b), omega=omega,
                  n_pre=n_pre, n_post=n_post).numpy()
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 1e-12, f"graph V-cycle departs from the reference algebra: {rel:.3e}"


def test_vcycle_on_conjugate_interface(sink, rng):
    """The coefficient jump is where coarsening is hardest, so check there too."""
    levels = build_hierarchy(sink)
    b = rng.standard_normal(sink.shape[0])
    want = reference_vcycle(levels, b, 0.7, 2, 2)
    got = v_cycle(levels, torch.as_tensor(b)).numpy()
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-12


def test_vcycle_on_curved_interface(bore, rng):
    levels = build_hierarchy(bore)
    b = rng.standard_normal(bore.shape[0])
    want = reference_vcycle(levels, b, 0.7, 2, 2)
    got = v_cycle(levels, torch.as_tensor(b)).numpy()
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-12


def test_vcycle_is_symmetric(cavity, rng):
    """<Bx,y> == <x,By>. If the preconditioner is not symmetric, CG is not valid."""
    levels = build_hierarchy(cavity)
    x = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    y = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    Bx, By = v_cycle(levels, x), v_cycle(levels, y)
    a, b = torch.dot(Bx, y).item(), torch.dot(x, By).item()
    assert abs(a - b) / max(abs(a), 1e-30) < 1e-12


def test_vcycle_is_positive_definite(cavity, rng):
    levels = build_hierarchy(cavity)
    for _ in range(5):
        x = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
        assert torch.dot(x, v_cycle(levels, x)).item() > 0


def test_vcycle_reduces_error(cavity, rng):
    """One cycle must actually be a useful correction, not just well-formed."""
    levels = build_hierarchy(cavity)
    A = levels[0].A
    x_true = torch.as_tensor(rng.standard_normal(cavity.shape[0]))
    b = A(x_true)
    e0 = torch.linalg.vector_norm(b - A(torch.zeros_like(b)))
    e1 = torch.linalg.vector_norm(b - A(v_cycle(levels, b)))
    assert e1 < 0.5 * e0, f"residual only fell {e0.item():.3e} -> {e1.item():.3e}"
