#!/bin/bash
#SBATCH --job-name=mwe_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=mwe_gpu_%j.out
# Runs every experiment twice over on one GPU node: the quick profile on the node's CPU and
# the paper profile on its GPU. Each run is checked against experiments/expected/<name>.<profile>.json, or rewrites
# it when EXPERIMENTS_MODE=refresh:
#
#     sbatch --account=<acct> --qos=<qos> run_experiments.sbatch
#     EXPERIMENTS_MODE=refresh sbatch --account=<acct> --qos=<qos> run_experiments.sbatch
#
# The conventions of run_gpu_check.sbatch hold here too. No account, partition or queue is
# set in this file. The directory comes from SLURM_SUBMIT_DIR, never from $0, which inside a
# job is Slurm's spool copy. Nothing is piped through tail.
#
# Install first, on a machine that can reach a package index: `uv sync --locked --extra dev`.
# Every command here is `uv run --frozen --offline`, so a job never re-resolves the lock and
# never downloads. It runs exactly the environment uv.lock describes, or fails.
#
# The paper profile is defined on one GPU. Without one the job stops at once, rather than
# running the paper sizes on a CPU and reporting that as the GPU result.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "FATAL: cannot enter ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
MODE=${EXPERIMENTS_MODE:-check}
case "$MODE" in
  check|refresh) ;;
  *) echo "FATAL: EXPERIMENTS_MODE must be check or refresh, not '$MODE'"; exit 1 ;;
esac
command -v uv >/dev/null || { echo "FATAL: uv is not on PATH"; exit 1; }
echo "commit: $(git rev-parse --short=12 HEAD 2>/dev/null || echo 'not a git checkout')"
echo "tree:   $(git status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "mode:   $MODE"
uv run --frozen --offline python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: torch sees no CUDA device, and the paper profile needs one"; exit 1; }
uv run --frozen --offline python -c 'import sys, torch, gnn4buoyancy
print("python:", sys.version.split()[0], " torch", torch.__version__, " cuda", torch.version.cuda,
      " gpu", torch.cuda.get_device_name(0), " gnn4buoyancy", gnn4buoyancy.__version__)'
rc=0
for profile in quick paper; do
  for name in vcycle_identity vcycle_spd hierarchy_rebuild stationary_vs_pcg \
              stationary_mismatch mixed_precision; do
    echo "--- $name, profile=$profile, mode=$MODE ---"
    uv run --frozen --offline gnn4buoyancy-experiment experiment=$name profile=$profile mode=$MODE \
      || rc=1
  done
done
echo "--- experiments exit=$rc ---"
exit $rc
