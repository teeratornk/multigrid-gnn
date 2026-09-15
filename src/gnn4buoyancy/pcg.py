"""Conjugate gradients preconditioned by one V-cycle, or by the `precondition` given.

The V-cycle is used as a preconditioner rather than as a standalone solver. As a
stationary iteration it converges only for a step length below 2/lambda_max(BA), which
has to be known or guessed, and the `stationary_mismatch` experiment shows what a wrong
one does. Inside conjugate gradients no step length is chosen. Conjugate gradients needs
the operator and the preconditioner to be symmetric positive definite, which is what
`vcycle_spd` checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .hierarchy import Level
from .vcycle import v_cycle


@dataclass
class SolveResult:
    x: torch.Tensor
    iterations: int
    residuals: list[float]
    converged: bool


def pcg(levels: list[Level], b: torch.Tensor, *, rtol: float = 1e-8, max_iter: int = 200,
        omega: float = 0.7, n_pre: int = 2, n_post: int = 2,
        x0: torch.Tensor | None = None, allow_unsymmetric: bool = False,
        operator=None, precondition=None) -> SolveResult:
    # `operator` replaces levels[0].A and `precondition` replaces one V-cycle on `levels`, on
    # every step including the first, whose direction is z0 = B r0.
    # The experiments need both: a cycle built on one operator applied to another, and an
    # fp32 cycle inside fp64 CG. Left as None they are exactly the original pair, and
    # tests/test_diagnostics.py checks that path bitwise against the previous code.
    # CG requires a SYMMETRIC preconditioner, and a V-cycle is symmetric only when the
    # pre- and post-smoothing counts match. They are separate config keys, so this is
    # reachable from the command line: `solver.n_pre=3` alone silently makes the
    # preconditioner unsymmetric. Measured on the 32x32 cavity, <Bx,y> vs <x,By> departs
    # by ~3e-2 at (3,1) against ~6e-16 at (2,2). CG may still converge on an easy system,
    # which is exactly what makes it dangerous: nothing looks wrong until it stalls on a
    # harder one. Refuse rather than return a result whose theory does not hold.
    if n_pre != n_post and not allow_unsymmetric:
        raise ValueError(
            f"n_pre={n_pre} != n_post={n_post} makes the V-cycle preconditioner "
            "unsymmetric, and conjugate gradients is only valid for a symmetric one. "
            "Use equal counts, or pass allow_unsymmetric=True if you are deliberately "
            "studying the effect (v_cycle itself supports unequal counts and is tested "
            "with them).")
    A = levels[0].A if operator is None else operator
    if precondition is None:
        def precondition(r):
            return v_cycle(levels, r, omega=omega, n_pre=n_pre, n_post=n_post)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A(x)
    b_norm = torch.linalg.vector_norm(b).item() or 1.0
    hist = [torch.linalg.vector_norm(r).item() / b_norm]
    if hist[0] <= rtol:
        return SolveResult(x, 0, hist, True)
    z = precondition(r)
    p, rz = z.clone(), torch.dot(r, z)
    done = 0  # completed iterations; the loop variable is not it on the breakdown and cap exits
    for k in range(1, max_iter + 1):
        Ap = A(p)
        pAp = torch.dot(p, Ap)
        if not (torch.isfinite(pAp) and pAp > 0 and torch.isfinite(rz)):
            # Breakdown: the recurrence has underflowed or the curvature is not positive, which
            # happens when rtol sits below the working precision. Stop with the true residual
            # of the last good iterate instead of dividing 0 by 0 into NaN.
            break
        alpha = rz / pAp
        x = x + alpha * p
        done = k
        r = r - alpha * Ap
        rel = torch.linalg.vector_norm(r).item() / b_norm
        hist.append(rel)
        if rel <= rtol:
            # NEVER return converged on the recurrence alone. `r` is updated by
            # r <- r - alpha*Ap and drifts from the true b - Ax. Unchecked, this reported
            # 5e-9 "converged" on the shipped float32 path while the true relative
            # residual was 7.6e-5, a factor of 7899, having never met rtol. It also bites
            # in float64 as the coefficient jump grows. Recompute once, report the TRUE
            # value, and restart the recurrence from it rather than stopping on a number
            # that is not the residual.
            r = b - A(x)
            rel = torch.linalg.vector_norm(r).item() / b_norm
            hist[-1] = rel
            if rel <= rtol:
                return SolveResult(x, done, hist, True)
            z = precondition(r)
            p, rz = z.clone(), torch.dot(r, z)
            continue
        z = precondition(r)
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    # Report the true residual on exit too, so a non-converged result is not described by
    # a drifted recurrence either. `done` counts the completed iterations, so `residuals`
    # keeps one entry per iteration plus the initial one on every exit.
    rel = (torch.linalg.vector_norm(b - A(x)).item() / b_norm)
    hist[-1] = rel
    return SolveResult(x, done, hist, rel <= rtol)
