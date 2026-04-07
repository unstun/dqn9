#!/usr/bin/env bash
# ============================================================================
# Auto pipeline: wait for 12 cnn-dqn trainings -> 24-parallel inference ->
# aggregate KPIs + paths to xlsx. Designed to run detached while the user
# sleeps; all progress goes to a single pipeline log.
#
# Launch (remote):
#   nohup bash ~/DQN9/scripts/auto_pipeline_cnn_dqn.sh \
#     > /tmp/auto_pipeline_cnn_dqn_boot.log 2>&1 < /dev/null &
#
# Phases:
#   A. wait for 12 train.py procs to exit (poll every 120s)
#   B. verify all 12 final models exist (<dir>/train_*/models/realmap_a/cnn-dqn.pt)
#   C. launch 24-parallel inference via launch_24par_cnn_dqn_infer.sh
#   D. wait for 24 infer.py procs to exit (poll every 60s)
#   E. run collect_cnn_dqn_to_xlsx.py -> runs20260408_dqn/aggregated_<ts>/
#   F. write PIPELINE_DONE_<ts> marker
# ============================================================================
set -uo pipefail

PROJ=/home/ubuntu/DQN9
ENV=ros2py310
CONDA=/home/ubuntu/miniconda3/bin/conda
RUNS_ROOT="${PROJ}/runs20260408_dqn"
LOG_DIR="${RUNS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PIPELINE_TS=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="${LOG_DIR}/auto_pipeline_${PIPELINE_TS}.log"

# redirect everything after this point to the pipeline log (append mode)
exec >> "${PIPELINE_LOG}" 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] ================================================================"
echo "[$(ts)] === cnn-dqn auto pipeline started (PIPELINE_TS=${PIPELINE_TS}) ==="
echo "[$(ts)] PROJ=${PROJ}"
echo "[$(ts)] RUNS_ROOT=${RUNS_ROOT}"
echo "[$(ts)] log=${PIPELINE_LOG}"

# ----------------------------------------------------------------------------
# Phase A: wait for 12 train.py procs to finish
# ----------------------------------------------------------------------------
echo ""
echo "[$(ts)] === Phase A: waiting for train.py procs to exit ==="

PREV_CKPT=-1
IDLE_WAIT_CYCLES=0
while true; do
  # count only training procs launched by our 8var/amdqfd 20260408 series
  N_TRAIN=$(pgrep -fc 'train\.py.*ablation_20260408' 2>/dev/null || true)
  N_TRAIN="${N_TRAIN//[!0-9]/}"
  : "${N_TRAIN:=0}"

  if [ "${N_TRAIN}" -eq 0 ]; then
    echo "[$(ts)] Phase A: train procs exited (N=0)"
    break
  fi

  CKPT=$(find "${RUNS_ROOT}" -name 'cnn-dqn_ep*.pt' 2>/dev/null | wc -l)
  if [ "${CKPT}" != "${PREV_CKPT}" ]; then
    echo "[$(ts)] Phase A: train_procs=${N_TRAIN}  total_ckpts=${CKPT}"
    PREV_CKPT="${CKPT}"
    IDLE_WAIT_CYCLES=0
  else
    IDLE_WAIT_CYCLES=$((IDLE_WAIT_CYCLES + 1))
    if [ "${IDLE_WAIT_CYCLES}" -ge 30 ]; then
      # 30 * 120s = 1h without new checkpoints -> warn
      echo "[$(ts)] Phase A: WARNING ${IDLE_WAIT_CYCLES} idle cycles with no new ckpt"
      IDLE_WAIT_CYCLES=0
    fi
  fi
  sleep 120
done

# ----------------------------------------------------------------------------
# Phase B: verify all 12 models exist
# ----------------------------------------------------------------------------
echo ""
echo "[$(ts)] === Phase B: verifying 12 final models ==="

MODEL_DIRS=(
  abl_8var_cnn_dqn
  abl_8var_cnn_dqn_duel
  abl_8var_cnn_dqn_munch
  abl_8var_cnn_dqn_munch_duel
  abl_amdqfd_cnn_dqn_noAM
  abl_amdqfd_cnn_dqn_noDQfD
  abl_amdqfd_cnn_dqn_duel_noAM
  abl_amdqfd_cnn_dqn_duel_noDQfD
  abl_amdqfd_cnn_dqn_munch_noAM
  abl_amdqfd_cnn_dqn_munch_noDQfD
  abl_amdqfd_cnn_dqn_munch_duel_noAM
  abl_amdqfd_cnn_dqn_munch_duel_noDQfD
)

MISSING=0
FOUND=0
for d in "${MODEL_DIRS[@]}"; do
  PT=$(ls "${RUNS_ROOT}/${d}"/train_*/models/realmap_a/cnn-dqn.pt 2>/dev/null | head -1)
  if [ -z "${PT}" ]; then
    echo "[$(ts)] Phase B: MISSING model for ${d}"
    MISSING=$((MISSING + 1))
  else
    SZ=$(stat -c%s "${PT}" 2>/dev/null || echo "?")
    echo "[$(ts)] Phase B: OK  ${d}  (${SZ} bytes)"
    FOUND=$((FOUND + 1))
  fi
done
echo "[$(ts)] Phase B: FOUND=${FOUND}  MISSING=${MISSING}"

if [ "${MISSING}" -gt 0 ]; then
  MARKER="${RUNS_ROOT}/PIPELINE_ABORTED_${PIPELINE_TS}"
  {
    echo "pipeline_ts=${PIPELINE_TS}"
    echo "aborted_at=$(ts)"
    echo "reason=models_missing"
    echo "missing_count=${MISSING}"
    echo "found_count=${FOUND}"
    echo "log=${PIPELINE_LOG}"
  } > "${MARKER}"
  echo "[$(ts)] === pipeline ABORTED (marker: ${MARKER}) ==="
  exit 1
fi

# ----------------------------------------------------------------------------
# Phase C: launch 24 parallel inference
# ----------------------------------------------------------------------------
echo ""
echo "[$(ts)] === Phase C: launching 24 parallel infer ==="
bash "${PROJ}/scripts/launch_24par_cnn_dqn_infer.sh" || {
  echo "[$(ts)] Phase C: launcher returned non-zero; continuing to wait anyway"
}
sleep 15

# ----------------------------------------------------------------------------
# Phase D: wait for 24 infer.py procs to finish
# ----------------------------------------------------------------------------
echo ""
echo "[$(ts)] === Phase D: waiting for 24 infer.py procs to exit ==="

PREV_KPI=-1
INFER_IDLE_CYCLES=0
while true; do
  N_INFER=$(pgrep -fc 'infer\.py.*ablation_20260408' 2>/dev/null || true)
  N_INFER="${N_INFER//[!0-9]/}"
  : "${N_INFER:=0}"

  if [ "${N_INFER}" -eq 0 ]; then
    echo "[$(ts)] Phase D: infer procs exited (N=0)"
    break
  fi

  KPI_DONE=$(find "${RUNS_ROOT}" -path '*/infer/*/table2_kpis_mean.csv' 2>/dev/null | wc -l)
  KPI_DONE_ALT=$(find "${RUNS_ROOT}" -name 'table2_kpis_mean.csv' 2>/dev/null | wc -l)
  if [ "${KPI_DONE_ALT}" != "${PREV_KPI}" ]; then
    echo "[$(ts)] Phase D: infer_procs=${N_INFER}  completed_mean_csv=${KPI_DONE_ALT}"
    PREV_KPI="${KPI_DONE_ALT}"
    INFER_IDLE_CYCLES=0
  else
    INFER_IDLE_CYCLES=$((INFER_IDLE_CYCLES + 1))
  fi
  sleep 60
done

# ----------------------------------------------------------------------------
# Phase E: aggregate to xlsx
# ----------------------------------------------------------------------------
echo ""
echo "[$(ts)] === Phase E: aggregating to xlsx ==="

AGG_DIR="${RUNS_ROOT}/aggregated_${PIPELINE_TS}"
"${CONDA}" run --cwd "${PROJ}" -n "${ENV}" python scripts/collect_cnn_dqn_to_xlsx.py \
  --runs-root "${RUNS_ROOT}" \
  --out-dir "${AGG_DIR}"
COLLECT_RC=$?
echo "[$(ts)] Phase E: collect script rc=${COLLECT_RC}"

# ----------------------------------------------------------------------------
# Phase F: write DONE marker
# ----------------------------------------------------------------------------
DONE_FILE="${RUNS_ROOT}/PIPELINE_DONE_${PIPELINE_TS}"
{
  echo "pipeline_ts=${PIPELINE_TS}"
  echo "finished_at=$(ts)"
  echo "log=${PIPELINE_LOG}"
  echo "aggregated_dir=${AGG_DIR}"
  echo "collect_rc=${COLLECT_RC}"
  echo "summary_xlsx=${AGG_DIR}/cnn_dqn_45_summary.xlsx"
  echo "raw_kpi_xlsx=${AGG_DIR}/cnn_dqn_45_raw_kpi.xlsx"
  echo "paths_xlsx=${AGG_DIR}/cnn_dqn_45_paths.xlsx"
} > "${DONE_FILE}"

echo ""
echo "[$(ts)] ================================================================"
echo "[$(ts)] === pipeline DONE (marker: ${DONE_FILE}) ==="
echo "[$(ts)] ================================================================"
