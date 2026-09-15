"""Hydra entry point: `gnn4buoyancy case=heat_sink solver=graph_mg case.params.n=128`."""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .cases import build_case
from .hierarchy import build_hierarchy
from .pcg import pcg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # NumPy too: pyamg's smoothed aggregation draws its spectral-radius start vector from NumPy's
    # global state, so without this two runs of one config differ in their last digits.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dtype = getattr(torch, cfg.dtype)
    A = build_case(cfg.case.name, **OmegaConf.to_container(cfg.case.params, resolve=True))
    print(f"case={cfg.case.name}  n={A.shape[0]}  nnz={A.nnz}")

    t0 = time.perf_counter()
    levels = build_hierarchy(A, max_levels=cfg.solver.max_levels, dtype=dtype,
                             device=cfg.device, use_sparse_mm=cfg.solver.use_sparse_mm)
    t_setup = time.perf_counter() - t0
    print(f"hierarchy: {len(levels)} levels, sizes "
          f"{[lv.A.shape[0] for lv in levels]}, setup {t_setup:.3f}s")

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    b = torch.randn(A.shape[0], generator=g, dtype=dtype).to(cfg.device)

    t0 = time.perf_counter()
    res = pcg(levels, b, rtol=cfg.solver.rtol, max_iter=cfg.solver.max_iter,
              omega=cfg.solver.omega, n_pre=cfg.solver.n_pre, n_post=cfg.solver.n_post)
    t_solve = time.perf_counter() - t0
    print(f"solve: {res.iterations} iterations, relative residual {res.residuals[-1]:.3e}, "
          f"{'converged' if res.converged else 'NOT CONVERGED'}, {t_solve:.3f}s")

    # Standard JSON only, as the experiments' writer: a NaN fails here instead of in a reader.
    Path(cfg.out).write_text(json.dumps(dict(
        config=OmegaConf.to_container(cfg, resolve=True),
        n=int(A.shape[0]), nnz=int(A.nnz), levels=len(levels),
        level_sizes=[int(lv.A.shape[0]) for lv in levels],
        iterations=res.iterations, converged=res.converged,
        residuals=res.residuals, setup_s=t_setup, solve_s=t_solve), indent=2, allow_nan=False))
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
