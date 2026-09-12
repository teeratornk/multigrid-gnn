"""The claim that a sparse apply IS message passing is exact, so test it exactly."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from gnn4buoyancy import MessagePassingOperator


@pytest.mark.parametrize("use_sparse_mm", [False, True])
def test_matches_scipy_to_roundoff(cavity, rng, use_sparse_mm):
    op = MessagePassingOperator(cavity, use_sparse_mm=use_sparse_mm)
    x = rng.standard_normal(cavity.shape[1])
    got = op(torch.as_tensor(x)).numpy()
    want = cavity @ x
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-14


def test_explicit_sum_over_neighbours(rng):
    """Check the identity row by row, not just in norm."""
    A = sp.csr_matrix(rng.standard_normal((12, 12)) * (rng.random((12, 12)) < 0.3))
    op = MessagePassingOperator(A)
    x = rng.standard_normal(12)
    got = op(torch.as_tensor(x)).numpy()
    for i in range(12):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        expect = sum(A.data[k] * x[A.indices[k]] for k in range(lo, hi))
        assert abs(got[i] - expect) < 1e-13, f"row {i}"


def test_both_backends_agree(cavity, rng):
    x = torch.as_tensor(rng.standard_normal(cavity.shape[1]))
    a = MessagePassingOperator(cavity, use_sparse_mm=False)(x)
    b = MessagePassingOperator(cavity, use_sparse_mm=True)(x)
    assert torch.allclose(a, b, rtol=0, atol=1e-13)


def test_linearity(cavity, rng):
    op = MessagePassingOperator(cavity)
    x = torch.as_tensor(rng.standard_normal(cavity.shape[1]))
    y = torch.as_tensor(rng.standard_normal(cavity.shape[1]))
    assert torch.allclose(op(2.5 * x + y), 2.5 * op(x) + op(y), atol=1e-12)


def test_nothing_is_trainable(cavity):
    """Weights are the assembled operator. If any of this required grad the paper's
    central claim, that nothing is trained, would be false."""
    op = MessagePassingOperator(cavity)
    assert list(op.parameters()) == []
    assert not any(b.requires_grad for b in op.buffers())


def test_roundtrip_to_scipy(cavity):
    back = MessagePassingOperator(cavity).to_scipy()
    assert abs(back - cavity).max() < 1e-14
