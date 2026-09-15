#!/bin/bash
#SBATCH --job-name=mwe_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=mwe_gpu_%j.out
# Runs the full test suite where a GPU exists and keeps the output as the evidence for the
# README's GPU claim. The eight device tests skip on a node without a usable CUDA device, and a
# green suite with them skipped proves nothing, so the job refuses to run in that case.
#
# No --account, --partition or --qos is set here. Those are site names, they mean nothing on
# another cluster, and they do not belong in a published file. Pass whatever your site needs
# on the command line, where it overrides these directives:
#
#     sbatch --account=<acct> --qos=<qos> run_gpu_check.sbatch
#
# Rules this script keeps:
#   * The directory comes from SLURM_SUBMIT_DIR, never from $0: Slurm runs a copy of the script
#     from its own spool directory, so $0 does not point at the checkout.
#   * pytest output is not truncated. The "N passed" summary is what this log exists to record.
#   * Every command is `uv run --frozen --offline` after `uv sync --locked --extra dev` on a
#     machine that reaches a package index, so the job runs the locked environment or fails.
#   * The exit status combines pytest and the GPU CLI step, so a failing CLI cannot hide behind
#     a green suite.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "FATAL: cannot enter ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
command -v uv >/dev/null || { echo "FATAL: uv is not on PATH"; exit 1; }
[ -d .venv ] || { echo "FATAL: no .venv here ($PWD); run 'uv sync --locked --extra dev' first"; exit 1; }
echo "host: $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "commit: $(git rev-parse --short=12 HEAD 2>/dev/null || echo 'not a git checkout')"
echo "tree:   $(git status --porcelain 2>/dev/null | wc -l) file(s) modified"
uv run --frozen --offline python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: torch sees no CUDA device; the device tests would skip and this log would prove nothing"; exit 1; }
echo "python: $(uv run --frozen --offline python -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))')"
echo "--- full test suite, no truncation ---"
uv run --frozen --offline --extra dev pytest tests/ -p no:warnings -rs
rc=$?
echo "--- pytest exit=$rc ---"
echo "--- CLI on GPU ---"
# outputs/gpu_check is overwritten on every run; the log above is the record. The CLI must exit 0
# and report a converged solve; an empty or unconverged run is a failure, not a quiet log.
cli_out=$(uv run --frozen --offline gnn4buoyancy device=cuda case=heat_sink hydra.run.dir=outputs/gpu_check 2>&1)
cli_rc=$?
printf '%s\n' "$cli_out" | grep -E "case=|hierarchy|solve:"
if [ "$cli_rc" -eq 0 ] && ! printf '%s\n' "$cli_out" | grep -qE "^solve: .*, converged,"; then
  echo "FATAL: the CLI did not report a converged solve"; cli_rc=1
fi
echo "--- cli exit=$cli_rc ---"
if [ "$rc" -eq 0 ] && [ "$cli_rc" -eq 0 ]; then echo "done."; else echo "FAILED: pytest exit=$rc, cli exit=$cli_rc"; fi
exit $(( rc != 0 ? rc : cli_rc ))
