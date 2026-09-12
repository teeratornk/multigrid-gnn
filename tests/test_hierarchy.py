"""Properties the multigrid hierarchy must have for the V-cycle to be meaningful."""
import numpy as np
import torch

from gnn4buoyancy import build_hierarchy


def test_levels_coarsen_strictly(cavity):
    lv = build_hierarchy(cavity)
    sizes = [l.A.shape[0] for l in lv]
    assert sizes[0] == cavity.shape[0]
    assert all(a > b for a, b in zip(sizes, sizes[1:])), sizes


def test_restriction_is_prolongation_transpose(cavity):
    """Smoothed aggregation sets R = P^T. If it does not, the V-cycle is not
    symmetric and using it to precondition CG is invalid."""
    for k, l in enumerate(build_hierarchy(cavity)[:-1]):
        R, P = l.R.to_scipy(), l.P.to_scipy()
        assert abs(R - P.T).max() / abs(P).max() < 1e-12, f"level {k}"


def test_transfer_shapes_chain(cavity):
    lv = build_hierarchy(cavity)
    for k in range(len(lv) - 1):
        assert lv[k].R.shape == (lv[k + 1].A.shape[0], lv[k].A.shape[0])
        assert lv[k].P.shape == (lv[k].A.shape[0], lv[k + 1].A.shape[0])


def test_diagonal_inverse_matches(cavity):
    for k, l in enumerate(build_hierarchy(cavity)):
        d = l.A.to_scipy().diagonal()
        assert np.allclose(l.diag_inv.numpy(), 1.0 / d, rtol=1e-13), f"level {k}"


def test_max_levels_is_respected(cavity):
    assert len(build_hierarchy(cavity, max_levels=2)) <= 2


def test_coarse_operator_is_the_galerkin_product(cavity):
    """A_{k+1} must equal R_k A_k P_k.

    This is the one hierarchy property the V-cycle identity test cannot see. That test
    builds its numpy reference from the same Level objects, so it is independent of
    vcycle.py but shares its entire input with hierarchy.py: permute the coarse operators
    while leaving R and P alone and the cycle still "matches its reference" even though a
    single V-cycle then amplifies the residual by 65x on engine_bore. Anchor the coarse
    operator to a scipy triple product computed from the fine level instead.
    """
    levels = build_hierarchy(cavity)
    for k in range(len(levels) - 1):
        A_k = levels[k].A.to_scipy()
        R, P = levels[k].R.to_scipy(), levels[k].P.to_scipy()
        want = (R @ A_k @ P).toarray()
        got = levels[k + 1].A.to_scipy().toarray()
        denom = max(abs(want).max(), 1e-30)
        assert abs(got - want).max() / denom < 1e-10, f"level {k+1} is not R A P"
