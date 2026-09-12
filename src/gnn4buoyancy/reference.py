"""The V-cycle restated in scipy, on the hierarchy the graph object holds.

This is the reference the identity experiment measures against. It reads the level
operators back out of the message-passing layers and runs the same algebra: weighted
Jacobi with the same omega and sweep counts, and the coarsest level relaxed rather than
solved. Nothing is rebuilt, so a difference can only come from the execution.

It is the cycle in tests/test_vcycle.py, moved here so an installed package can run it.
tests/test_diagnostics.py checks the two agree bit for bit.
"""
from __future__ import annotations

import numpy as np

from .hierarchy import Level


def reference_vcycle(levels: list[Level], b: np.ndarray, *, omega: float, n_pre: int,
                     n_post: int) -> np.ndarray:
    A = [l.A.to_scipy() for l in levels]
    dinv = [l.diag_inv.cpu().numpy() for l in levels]
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
