"""Operators for the example cases.

All three are the symmetric positive definite operator of a projection step, on geometries
small enough to run on a laptop CPU. conjugate_cavity is the constant-coefficient pressure
Poisson operator; the other two carry a conductivity jump across a solid-fluid interface,
which is what makes the coarsening hard. The engine geometry of the
applied section is not included here; engine_bore is a parametric stand-in for its
curved interface.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def conjugate_cavity(n: int) -> sp.csr_matrix:
    """Pressure Laplacian on a square cavity, 5-point, Dirichlet."""
    import pyamg
    return sp.csr_matrix(pyamg.gallery.poisson((n, n), format="csr"), dtype=np.float64)


def heat_sink(n: int, *, fins: int = 4, contrast: float = 1e3) -> sp.csr_matrix:
    """Conjugate heat-sink diffusion: a fin array of high conductivity in a fluid.

    The interface is where the conjugate problem is hard, so the coefficient jump
    is what makes this a more demanding test than the uniform cavity.

    Assembled as a variable-coefficient five-point Laplacian with harmonic face
    averaging, which is the standard treatment for a jump and, unlike scaling a
    constant-coefficient operator by a diagonal, keeps the result symmetric. A
    non-symmetric operator here would not be a diffusion problem at all and would
    break the conjugate-gradient solve; tests/test_cases.py checks both properties.
    """
    k = np.ones((n, n))
    lo, hi = n // 4, 3 * n // 4
    for f in range(fins):                             # vertical fins, evenly spaced
        c = int((f + 0.5) * n / fins)
        k[lo:hi, c] = contrast

    return _variable_coefficient_laplacian(k)


def _variable_coefficient_laplacian(k: np.ndarray) -> sp.csr_matrix:
    """Five-point Laplacian for a per-cell conductivity field.

    Harmonic averaging on the faces, which is the standard treatment for a jump,
    and Dirichlet closure at the domain edge. Assembling this way rather than
    scaling a constant-coefficient operator by a diagonal is what keeps the result
    symmetric; tests/test_cases.py checks that for every case.
    """
    n = k.shape[0]
    idx = lambda i, j: i * n + j
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            p, diag = idx(i, j), 0.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n:
                    kf = 2.0 * k[i, j] * k[a, b] / (k[i, j] + k[a, b])   # harmonic
                    rows.append(p); cols.append(idx(a, b)); vals.append(-kf)
                else:
                    kf = k[i, j]                       # Dirichlet closure
                diag += kf
            rows.append(p); cols.append(p); vals.append(diag)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n * n, n * n)).tocsr()
    return sp.csr_matrix((A + A.T) * 0.5)              # symmetric to round-off


def engine_bore(n: int, *, bore_frac: float = 0.42, contrast: float = 1e3,
                liner_frac: float = 0.06, liner_contrast: float = 3e2) -> sp.csr_matrix:
    """Mid-plane section of a single-cylinder engine: a gas bore inside metal.

    A representative section, generated parametrically. It is NOT the
    geometry the paper's engine results were computed on; that mesh is not part of
    this example. What it reproduces is the feature that makes the applied case
    hard for the coarsening, and which neither of the other two cases has: a
    *curved* solid-fluid interface. On a structured grid the interface staircases,
    so the coefficient jump lands on a ragged boundary rather than a straight line.

    Three regions: the bore (low conductivity gas), a thin liner, and the
    surrounding block (high conductivity metal).
    """
    c = (n - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt((ii - c) ** 2 + (jj - c) ** 2) / c        # 0 at centre, 1 at edge
    k = np.full((n, n), contrast)                          # metal block
    k[r <= bore_frac + liner_frac] = liner_contrast        # liner
    k[r <= bore_frac] = 1.0                                # gas bore
    return _variable_coefficient_laplacian(k)


CASES = {"conjugate_cavity": conjugate_cavity, "heat_sink": heat_sink,
         "engine_bore": engine_bore}


def build_case(name: str, **kw) -> sp.csr_matrix:
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; available: {sorted(CASES)}")
    return CASES[name](**kw)
