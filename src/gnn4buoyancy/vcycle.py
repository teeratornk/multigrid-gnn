"""The V-cycle, as a graph U-Net.

Down the levels: smooth, form the residual, restrict. Up: prolong the coarse
correction, smooth again. Every one of those is a message-passing application on
that level's graph, so the whole cycle is one kind of operation on a sequence of
graphs. The coarsest level is relaxed by the same smoother, not solved directly.
"""
from __future__ import annotations

import torch

from .hierarchy import Level


def jacobi(A, diag_inv, x, b, n, omega):
    for _ in range(n):
        x = x + omega * diag_inv * (b - A(x))
    return x


def v_cycle(levels: list[Level], b: torch.Tensor, x: torch.Tensor | None = None, *,
            omega: float = 0.7, n_pre: int = 2, n_post: int = 2, level: int = 0):
    lv = levels[level]
    if x is None:
        x = torch.zeros_like(b)
    x = jacobi(lv.A, lv.diag_inv, x, b, n_pre, omega)
    if level == len(levels) - 1:
        return x
    r_coarse = lv.R(b - lv.A(x))
    e = v_cycle(levels, r_coarse, None, omega=omega, n_pre=n_pre,
                n_post=n_post, level=level + 1)
    x = x + lv.P(e)
    return jacobi(lv.A, lv.diag_inv, x, b, n_post, omega)
