"""A sparse operator applied as message passing.

The identity the paper rests on is exact, not an approximation:

    (A x)_i = sum_{j in N(i)} A_ij x_j

so a row of a sparse matrix IS a message-passing neighbourhood with the matrix
entry as the edge weight. Nothing here is learned; the weights are the assembled
finite-element operator.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class MessagePassingOperator(nn.Module):
    """Applies a fixed sparse operator by gather, weight, scatter-add.

    Deliberately written as the explicit three-step form rather than a call into
    torch.sparse, so a reader can see that the layer performs exactly the sum
    above. `use_sparse_mm=True` swaps in the vendor kernel; both paths are tested
    against each other and against scipy.
    """

    def __init__(self, A: sp.spmatrix, *, dtype=torch.float64, use_sparse_mm: bool = False):
        super().__init__()
        A = sp.coo_matrix(A)
        self.shape = A.shape
        self.use_sparse_mm = bool(use_sparse_mm)
        self.register_buffer("target", torch.as_tensor(A.row.astype(np.int64)))
        self.register_buffer("source", torch.as_tensor(A.col.astype(np.int64)))
        self.register_buffer("weight", torch.as_tensor(A.data.astype(np.float64)).to(dtype))
        if self.use_sparse_mm:
            idx = torch.stack([self.target, self.source])
            self.register_buffer(
                "_A", torch.sparse_coo_tensor(idx, self.weight, A.shape).coalesce())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both paths must agree on what is legal. torch.sparse.mm REFUSES a dtype
        # mismatch, while gather/scatter silently promoted: an fp32 operator applied to
        # an fp64 vector returned an fp64 tensor carrying fp32 accuracy, 6e-5 wrong
        # against a direct solve while the solver reported 3e-11 and "converged". Since
        # this class claims the two backends are tested against each other, reject it
        # here as well rather than letting the answer depend on which one is selected.
        if x.dtype != self.weight.dtype:
            raise TypeError(
                f"operator dtype {self.weight.dtype} but input dtype {x.dtype}. "
                "Mixing them silently degrades the result to the lower precision while "
                "returning the higher one; build the hierarchy at the dtype you intend.")
        if self.use_sparse_mm:
            return torch.sparse.mm(self._A, x.unsqueeze(1)).squeeze(1)
        messages = self.weight * x.index_select(0, self.source)   # A_ij x_j
        out = torch.zeros(self.shape[0], dtype=x.dtype, device=x.device)
        return out.index_add_(0, self.target, messages)           # sum over j in N(i)

    def to_scipy(self) -> sp.csr_matrix:
        """Recover the operator, so tests can compare against scipy directly."""
        return sp.coo_matrix(
            (self.weight.detach().cpu().numpy(),
             (self.target.cpu().numpy(), self.source.cpu().numpy())),
            shape=self.shape).tocsr()
