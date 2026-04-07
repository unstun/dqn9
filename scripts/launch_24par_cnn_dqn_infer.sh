#!/usr/bin/env bash
# ============================================================================
# Phase 9: Launch 24 parallel cnn-dqn inference profiles in background
# ----------------------------------------------------------------------------
# 24 = 4 variants (vanilla / duel / munch / munch_duel)
#    × 3 conditions (full / noAM / noDQfD)
#    × 2 distances (short 6-14m / long >=18m)
#
# Each profile is detached via setsid so the launcher returns immediately
# after spawning all 24 children. PYTHONUNBUFFERED=1 + python -u attempts to
# flush log files in real time (conda-run still buffers a bit; rely on GPU
# metrics + output CSVs as the authoritative progress signal).
# ============================================================================
set -euo pipefail

PROJ=/home/ubuntu/DQN9
ENV=ros2py310
CONDA=/home/ubuntu/miniconda3/bin/conda
RUNS_ROOT=runs20260408_dqn
LOG_DIR="${PROJ}/${RUNS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PROFILES=(
  # ---- Full condition (reuse 8var base models, AM+DQfD both ON) ----
  ablation_20260408_amdqfd_cnn_dqn_infer_full_sr_short
  ablation_20260408_amdqfd_cnn_dqn_infer_full_sr_long
  ablation_20260408_amdqfd_cnn_dqn_duel_infer_full_sr_short
  ablation_20260408_amdqfd_cnn_dqn_duel_infer_full_sr_long
  ablation_20260408_amdqfd_cnn_dqn_munch_infer_full_sr_short
  ablation_20260408_amdqfd_cnn_dqn_munch_infer_full_sr_long
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_infer_full_sr_short
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_infer_full_sr_long
  # ---- noAM condition (action shield OFF) ----
  ablation_20260408_amdqfd_cnn_dqn_infer_noAM_sr_short
  ablation_20260408_amdqfd_cnn_dqn_infer_noAM_sr_long
  ablation_20260408_amdqfd_cnn_dqn_duel_infer_noAM_sr_short
  ablation_20260408_amdqfd_cnn_dqn_duel_infer_noAM_sr_long
  ablation_20260408_amdqfd_cnn_dqn_munch_infer_noAM_sr_short
  ablation_20260408_amdqfd_cnn_dqn_munch_infer_noAM_sr_long
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_infer_noAM_sr_short
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_infer_noAM_sr_long
  # ---- noDQfD condition (demo prefill OFF) ----
  ablation_20260408_amdqfd_cnn_dqn_infer_noDQfD_sr_short
  ablation_20260408_amdqfd_cnn_dqn_infer_noDQfD_sr_long
  ablation_20260408_amdqfd_cnn_dqn_duel_infer_noDQfD_sr_short
  ablation_20260408_amdqfd_cnn_dqn_duel_infer_noDQfD_sr_long
  ablation_20260408_amdqfd_cnn_dqn_munch_infer_noDQfD_sr_short
  ablation_20260408_amdqfd_cnn_dqn_munch_infer_noDQfD_sr_long
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_infer_noDQfD_sr_short
  ablation_20260408_amdqfd_cnn_dqn_munch_duel_infer_noDQfD_sr_long
)

TS=$(date +%Y%m%d_%H%M%S)
echo "[$(date '+%F %T')] launching ${#PROFILES[@]} parallel cnn-dqn infer profiles (ts=${TS})"

cd "${PROJ}"
for p in "${PROFILES[@]}"; do
  LOG="${LOG_DIR}/infer_${p}_${TS}.log"
  setsid bash -c "PYTHONUNBUFFERED=1 \"${CONDA}\" run --cwd \"${PROJ}\" -n \"${ENV}\" \
    python -u infer.py --profile \"${p}\" --runs-root \"${RUNS_ROOT}\" \
    > \"${LOG}\" 2>&1 < /dev/null" &
  echo "  spawned pid=$! profile=${p}"
done

sleep 8
echo
echo "After 8s wait:"
echo "  infer.py procs: $(pgrep -fc 'infer\.py.*ablation_20260408' || echo 0)"
echo "  GPU memory   : $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "  GPU util     : $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
echo
echo "[$(date '+%F %T')] launcher exiting; 24 infer children continue in background."
