"""Measurements the experiments report.

Each function returns a number with one stated meaning. None of them decides pass or
fail; the experiment configs state the claims, and the numbers are recorded beside them.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import torch

from .hierarchy import Level


def relative_difference(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def symmetry_defect(apply, x: torch.Tensor, y: torch.Tensor) -> float:
    """|<Bx,y> - <x,By>| / (||Bx|| ||y||). Round-off for a symmetric B.

    Scaled by norms rather than by |<Bx,y>|. For random x and y that inner product can land
    near zero, and dividing by it turned the same unequal-count cycle into 1e-2 on one grid
    and 4.4 on another. The norm product cannot vanish for a nonzero B.
    """
    Bx = apply(x)
    a, c = torch.dot(Bx, y).item(), torch.dot(x, apply(y)).item()
    return abs(a - c) / (torch.linalg.vector_norm(Bx).item() * torch.linalg.vector_norm(y).item())


def min_rayleigh(apply, vectors: list[torch.Tensor]) -> float:
    """Smallest <x,Bx>/<x,x> over the given vectors. Positive for a positive definite B."""
    return min(torch.dot(v, apply(v)).item() / torch.dot(v, v).item() for v in vectors)


def transpose_defect(levels: list[Level]) -> float:
    """max over levels of max|R - P^T| / max|P|. Smoothed aggregation sets R = P^T."""
    worst = 0.0
    for lv in levels[:-1]:
        R, P = lv.R.to_scipy(), lv.P.to_scipy()
        worst = max(worst, float(abs(R - P.T).max() / abs(P).max()))
    return worst


def _arrays(levels: list[Level]) -> list[np.ndarray]:
    out = []
    for lv in levels:
        for op in (lv.A, lv.R, lv.P):
            if op is not None:
                out += [op.target, op.source, op.weight]
        out.append(lv.diag_inv)
    return [t.detach().cpu().numpy() for t in out]


def bitwise_equal(first: list[Level], second: list[Level]) -> bool:
    """Every index and weight the two hierarchies hold is the same, bit for bit.

    Compared as bytes, not with ==, which would call -0.0 equal to 0.0.
    """
    a, b = _arrays(first), _arrays(second)
    return len(a) == len(b) and all(
        x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        for x, y in zip(a, b))


def coarse_drift(first: list[Level], second: list[Level]) -> float | None:
    """max over coarse levels of max|A_1 - A_2| / max|A_2|. None if the shapes differ."""
    if [lv.A.shape for lv in first] != [lv.A.shape for lv in second]:
        return None
    worst = 0.0
    for a, b in zip(first[1:], second[1:]):
        A1, A2 = a.A.to_scipy(), b.A.to_scipy()
        worst = max(worst, float(abs(A1 - A2).max() / abs(A2).max()))
    return worst


def ritz_values(operator, precondition, v0: torch.Tensor, *, steps: int) -> np.ndarray:
    """Eigenvalue estimates of B A, from the Lanczos matrix preconditioned CG builds.

    CG preconditioned by an SPD B is Lanczos in the B inner product. Its step lengths
    alpha_j and ratios beta_j = <r_{j+1},z_{j+1}>/<r_j,z_j> fill the tridiagonal T with
    T_jj = 1/alpha_j + beta_{j-1}/alpha_{j-1} and T_j,j+1 = sqrt(beta_j)/alpha_j. Its
    eigenvalues converge to those of BA from the extremes inward, and always lie between
    them. The recursion stops early once the residual has fallen by the working
    precision, since the Krylov space has then stopped growing.
    """
    r = v0.clone()
    z = precondition(r)
    p, rz = z.clone(), torch.dot(r, z).item()
    floor = torch.finfo(v0.dtype).eps ** 2 * rz
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(steps):
        Ap = operator(p)
        alpha = rz / torch.dot(p, Ap).item()
        alphas.append(alpha)
        r = r - alpha * Ap
        z = precondition(r)
        rz_new = torch.dot(r, z).item()
        if not rz_new > floor:
            break
        betas.append(rz_new / rz)
        p = z + betas[-1] * p
        rz = rz_new
    k = len(alphas)
    diag = np.array([1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j else 0.0)
                     for j in range(k)])
    off = np.array([np.sqrt(betas[j]) / alphas[j] for j in range(k - 1)])
    return scipy.linalg.eigh_tridiagonal(diag, off, eigvals_only=True)


def stationary_radius(ritz_min: float, ritz_max: float, omega: float) -> float:
    """Spectral radius of I - omega B A, from the extreme eigenvalues of BA."""
    return max(abs(1.0 - omega * ritz_min), abs(1.0 - omega * ritz_max))


def convergence_factor(residuals: list[float], *, window: int) -> float:
    """Mean contraction per step over the last `window` steps of a residual history."""
    w = min(window, len(residuals) - 1)
    if w < 1:
        return 0.0
    return (residuals[-1] / residuals[-1 - w]) ** (1.0 / w)
