"""Turn an assembled operator into a stack of message-passing layers.

The hierarchy comes from smoothed aggregation (PyAMG) and is then frozen: the
level operators, restriction and prolongation become fixed edge weights. The
setup runs once, outside anything differentiated.

Rebuilds from the same matrix are not bit-reproducible unless NumPy is seeded first.
pyamg estimates the spectral radius used to smooth the prolongator from a random start
vector drawn from NumPy's global RNG, so two unseeded builds give coarse operators that
differ by 1e-4 to 2e-2 relative on the example cases. Seed NumPy before the build, as the
experiments do, and a rebuild is identical bit for bit (experiment hierarchy_rebuild).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .layers import MessagePassingOperator


@dataclass
class Level:
    A: MessagePassingOperator
    diag_inv: torch.Tensor
    R: MessagePassingOperator | None = None   # fine -> coarse
    P: MessagePassingOperator | None = None   # coarse -> fine


def build_hierarchy(A: sp.spmatrix, *, max_levels: int = 10, dtype=torch.float64,
                    device="cpu", use_sparse_mm: bool = False) -> list[Level]:
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A), max_levels=max_levels)
    levels: list[Level] = []
    for k, lvl in enumerate(ml.levels):
        Ak = sp.csr_matrix(lvl.A)
        d = Ak.diagonal()
        if np.any(d == 0):
            raise ValueError(f"level {k} has a zero diagonal; Jacobi smoothing undefined")
        lv = Level(
            A=MessagePassingOperator(Ak, dtype=dtype, use_sparse_mm=use_sparse_mm).to(device),
            diag_inv=torch.as_tensor(1.0 / d, dtype=dtype, device=device),
        )
        if k < len(ml.levels) - 1:
            lv.R = MessagePassingOperator(sp.csr_matrix(lvl.R), dtype=dtype,
                                          use_sparse_mm=use_sparse_mm).to(device)
            lv.P = MessagePassingOperator(sp.csr_matrix(lvl.P), dtype=dtype,
                                          use_sparse_mm=use_sparse_mm).to(device)
        levels.append(lv)
    return levels


def cast_levels(levels: list[Level], dtype) -> list[Level]:
    """The same hierarchy at another precision, on the device it already occupies.

    The frozen weights are cast, not rebuilt, so precision is the only thing that
    changes. A rebuild would rerun smoothed aggregation, whose spectral-radius estimate
    starts from NumPy's global RNG. Entries keep their stored order, so casting to the
    dtype already held gives the same cycle bit for bit, and casting a seeded fp64 build
    gives exactly the fp32 build from the same seed.
    """
    def cast(op: MessagePassingOperator) -> MessagePassingOperator:
        coo = sp.coo_matrix((op.weight.detach().cpu().numpy(),
                             (op.target.cpu().numpy(), op.source.cpu().numpy())),
                            shape=op.shape)
        return MessagePassingOperator(coo, dtype=dtype, use_sparse_mm=op.use_sparse_mm
                                      ).to(op.weight.device)

    out: list[Level] = []
    for lv in levels:
        e = Level(A=cast(lv.A), diag_inv=lv.diag_inv.to(dtype))
        if lv.R is not None:
            e.R, e.P = cast(lv.R), cast(lv.P)
        out.append(e)
    return out
