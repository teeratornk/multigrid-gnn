"""The V-cycle as a standalone solver: x <- x + omega * B(b - A x).

The package never solves this way. The iteration is here so the experiments can show
why. Its error propagator is E = I - omega B A, which contracts only when
omega * lambda_max(BA) < 2, and whose rate is fixed by the two extreme eigenvalues of BA.
Conjugate gradients with the same B needs no step length and adapts to the whole
spectrum.

The residual is recomputed as b - Ax at every step. There is no recurrence to drift, so
the history is the true one, the same standard pcg.py holds itself to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class StationaryResult:
    x: torch.Tensor
    iterations: int
    residuals: list[float]
    converged: bool
    diverged: bool


def stationary(operator, precondition, b: torch.Tensor, *, omega: float = 1.0,
               rtol: float = 1e-8, max_iter: int = 200,
               divergence_ratio: float = 1e8) -> StationaryResult:
    """Iterate from zero until the relative residual meets rtol or passes divergence_ratio.

    `divergence_ratio` bounds the relative residual, so a diverging run stops before it
    overflows rather than after, and the history never holds inf or nan.
    """
    x = torch.zeros_like(b)
    r = b - operator(x)
    b_norm = torch.linalg.vector_norm(b).item() or 1.0
    hist = [torch.linalg.vector_norm(r).item() / b_norm]
    if hist[0] <= rtol:
        return StationaryResult(x, 0, hist, True, False)
    for k in range(1, max_iter + 1):
        x = x + omega * precondition(r)
        r = b - operator(x)
        rel = torch.linalg.vector_norm(r).item() / b_norm
        if not math.isfinite(rel):
            return StationaryResult(x, k, hist, False, True)
        hist.append(rel)
        if rel > divergence_ratio:
            return StationaryResult(x, k, hist, False, True)
        if rel <= rtol:
            return StationaryResult(x, k, hist, True, False)
    return StationaryResult(x, max_iter, hist, False, False)
