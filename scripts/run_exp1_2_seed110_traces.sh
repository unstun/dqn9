#!/bin/bash
# EXP1.2: 核心对比 CNN-DDQN+MD vs RRT* vs LO-HA* (seed 110, 带 traces)
# seed 110, 拆为 10 sub-seed (110-119) x 5 runs = 50 runs
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
MAX_PARALLEL=10
LOGDIR="$PROJ/runs/exp1_2_logs"
MASTER_LOG="$LOGDIR/master.log"

mkdir -p "$LOGDIR"
log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$MASTER_LOG"; }

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

log "===== EXP1.2: Core comparison (seed 110, 50 runs, with traces) ====="

profiles=(final_t10_sr_long final_t10_sr_short)
base_seed=110
sub_count=10
runs_per=5

for profile in "${profiles[@]}"; do
  out_name="${profile}.2"
  for ((s=0; s<sub_count; s++)); do
    seed=$((base_seed + s))
    wait_for_slot
    logf="$LOGDIR/${profile##final_t10_}_s${seed}.log"
    log "  $profile seed=$seed runs=$runs_per -> $out_name"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" \
        --seed "$seed" --runs "$runs_per" \
        --save-traces --forest-baseline-save-traces \
        --out "$out_name" \
      > "$logf" 2>&1 &
  done
done

log "All jobs dispatched, waiting..."
wait
log "EXP1.2 DONE. Results: runs/final_t10_sr_{long,short}.2/"
