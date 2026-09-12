"""fp32/fp64 and CPU/GPU paths must not silently disagree."""
import numpy as np
import pytest
import torch

from gnn4buoyancy import build_hierarchy, pcg, v_cycle
from gnn4buoyancy.hierarchy import Level
from gnn4buoyancy.layers import MessagePassingOperator


def _cast_levels(levels, dtype):
    """The same hierarchy at another precision, so only precision varies."""
    out = []
    for l in levels:
        e = Level(A=MessagePassingOperator(l.A.to_scipy(), dtype=dtype),
                  diag_inv=l.diag_inv.to(dtype))
        if l.R is not None:
            e.R = MessagePassingOperator(l.R.to_scipy(), dtype=dtype)
            e.P = MessagePassingOperator(l.P.to_scipy(), dtype=dtype)
        out.append(e)
    return out


def test_float32_tracks_float64(cavity, rng):
    """Single precision must give the same cycle to within its own precision.

    The bound is fp32 epsilon (about 1e-7) amplified by the operator condition
    number (about 1e3 for this Poisson at these sizes), so a few times 1e-4 is
    the expected size of the gap. Asserting anything much tighter would be
    testing round-off luck rather than that the fp32 path is the same algorithm.
    """
    b = rng.standard_normal(cavity.shape[0])
    lv64 = build_hierarchy(cavity, dtype=torch.float64)
    # Cast THIS hierarchy rather than building a second one. Smoothed aggregation
    # is not reproducible across rebuilds (see README): two builds of the same
    # matrix can differ by ~1e-3 in the coarse operators, which is larger than the
    # precision effect being measured and made this test flaky.
    lv32 = _cast_levels(lv64, torch.float32)
    r64 = v_cycle(lv64, torch.as_tensor(b, dtype=torch.float64)).numpy()
    r32 = v_cycle(lv32, torch.as_tensor(b, dtype=torch.float32)).numpy()
    rel = np.linalg.norm(r32 - r64) / np.linalg.norm(r64)
    assert rel < 1e-4, f"fp32 departs from fp64 by {rel:.2e}, beyond round-off"
    assert rel > 0, "fp32 and fp64 identical, suggesting the dtype was ignored"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_cpu(cavity, rng):
    b = rng.standard_normal(cavity.shape[0])
    cpu = pcg(build_hierarchy(cavity, device="cpu"),
              torch.as_tensor(b), rtol=1e-10, max_iter=200)
    gpu = pcg(build_hierarchy(cavity, device="cuda"),
              torch.as_tensor(b).cuda(), rtol=1e-10, max_iter=200)
    assert cpu.iterations == gpu.iterations
    assert np.linalg.norm(gpu.x.cpu().numpy() - cpu.x.numpy()) / \
           np.linalg.norm(cpu.x.numpy()) < 1e-8
