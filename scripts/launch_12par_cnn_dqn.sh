#!/usr/bin/env bash
# ============================================================================
# Phase 8: Launch 12 parallel cnn-dqn training profiles in background
# ----------------------------------------------------------------------------
# Each profile is detached via nohup + setsid so the launcher returns
# immediately after spawning all 12 children. PYTHONUNBUFFERED=1 + python -u
# ensure log files flush in real time for monitoring.
# ============================================================================
set -euo pipefail

PROJ=/home/ubuntu/DQN9
ENV=ros2py310
CONDA=/home/ubuntu/miniconda3/bin/conda
RUNS_ROOT=runs20260408_dqn
LOG_DIR="${PROJ}/${RUNS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PROFILES=(
  ablation_20260408_8var_cnn_dqn
  ablation_20260408_8var_cnn_dqn_duel
  ablation_20260408_8var_cnn_dqn_munch
  ablation_20260408_8var_cnn_dqn_munch_duel
  ablation_20260408_amdqfd_cnn_dqn_noAM
  ablation_20260408_amdqfd_cnn_dqn_noDQfD
  ablation_20260408_amdqfd_cnn_dqn_duel_noAM
  ablation_20260408_amdqfd_cnn_dqn_duel_noDQfD
  ablation_20260408_amdqfd_cnn_dqn_munch_noAM
  ablation_20260408_amdqfd_cnn_dqn_munch_noDQfD
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_noAM
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_noDQfD
)

TS=$(date +%Y%m%d_%H%M%S)
echo "Launching ${#PROFILES[@]} parallel cnn-dqn profiles at ${TS}"

cd "${PROJ}"
for p in "${PROFILES[@]}"; do
  LOG="${LOG_DIR}/${p}_${TS}.log"
  setsid bash -c "PYTHONUNBUFFERED=1 \"${CONDA}\" run --cwd \"${PROJ}\" -n \"${ENV}\" \
    python -u train.py --profile \"${p}\" --runs-root \"${RUNS_ROOT}\" \
    > \"${LOG}\" 2>&1 < /dev/null" &
  echo "  spawned pid=$! profile=${p}"
done

wait_count=0
sleep 3
echo
echo "After 3s wait:"
echo "  train.py procs: $(pgrep -fc 'train\.py.*ablation_20260408' || echo 0)"
echo "  GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
echo "Launcher exiting; children continue in background."
