#!/bin/bash
# Seed 1000 测试：50 runs, 2 distances, 15 并行
set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/seed1000_logs
MASTER_LOG=$PROJ/seed1000_master.log

# 10 batch × 5 runs = 50 runs
SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)
RUNS=(5 5 5 5 5 5 5 5 5 5)

PROFILES=(
  mpc_md_kt02_sr_long
  mpc_md_kt02_sr_short
)

mkdir -p "$LOGS"
total_jobs=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
echo "$(date) START seed1000 test: $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee -a "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

started=0
for profile in "${PROFILES[@]}"; do
  short=$(echo "$profile" | sed 's/mpc_md_kt02_//')
  for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    runs=${RUNS[$i]}
    wait_for_slot
    log="$LOGS/${short}_s${seed}.log"
    echo "$(date) START: $short seed=$seed runs=$runs -> $log" >> "$MASTER_LOG"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" --seed "$seed" --runs "$runs" \
      > "$log" 2>&1 &
    started=$((started + 1))
  done
done

echo "$(date) Waiting for $started jobs to finish..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL SEED1000 TEST DONE" | tee -a "$MASTER_LOG"
