#!/bin/bash
#SBATCH --job-name=mwe_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=mwe_gpu_%j.out
# The two device tests skip on a CPU node, so run them where a card exists and keep the
# output as the evidence for the README's GPU claim.
#
# No --account, --partition or --qos is set here. Those are site names, they mean nothing
# on another cluster, and they do not belong in a file that is going to be published. Pass
# whatever your site needs on the command line, where it overrides these directives:
#
#     sbatch --account=<acct> --qos=<qos> run_gpu_check.sbatch
#
# Two things here are deliberate, and both were mistakes in the first version.
#
# The directory comes from SLURM_SUBMIT_DIR rather than being hardcoded. The example
# exists in two places (a standalone repo and a copy inside the AI4PDEs tree) and a
# hardcoded path silently runs the wrong one, or none: `cd` failing does not stop the
# script, because -e is not set and cannot be, since we want the summary printed even
# when pytest fails.
#
# Do NOT derive it from $0. Slurm copies the batch script into a per-job spool directory,
# so inside the job $0 is /cm/local/.../job<id>/slurm_script and `dirname $0` points at
# the spool, not the checkout. That was tried and failed with exit 127.
#
# pytest's output is NOT truncated. It was piped through `tail -4`, which cut the
# "N passed" summary and left only a wrapped progress line. That made the log unable to
# confirm how many tests ran, which is the one thing this artifact exists to record.
#
# Every command is `uv run --frozen --offline`, after `uv sync --locked --extra dev` on a
# machine that can reach a package index. The first version ran `.venv/bin/python`, which
# runs whatever the venv happens to hold, and that venv had drifted from uv.lock. `--frozen`
# never re-resolves the lock and `--offline` never downloads, so a job runs the locked
# environment or fails. Compute nodes here cannot reach an index in any case.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "FATAL: cannot enter ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
command -v uv >/dev/null || { echo "FATAL: uv is not on PATH"; exit 1; }
[ -d .venv ] || { echo "FATAL: no .venv here ($PWD); run 'uv sync --locked --extra dev' first"; exit 1; }
echo "host: $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit: $(git rev-parse --short=12 HEAD 2>/dev/null || echo 'not a git checkout')"
echo "tree:   $(git status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "python: $(uv run --frozen --offline python -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__)')"
echo "--- full test suite, no truncation ---"
uv run --frozen --offline --extra dev pytest tests/ -p no:warnings -rs
rc=$?
echo "--- pytest exit=$rc ---"
echo "--- CLI on GPU ---"
uv run --frozen --offline gnn4buoyancy device=cuda case=heat_sink hydra.run.dir=/tmp/gnn4buoyancy_gpu 2>&1 \
  | grep -E "case=|hierarchy|solve:"
echo "done."
exit $rc
