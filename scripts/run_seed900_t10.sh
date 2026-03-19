#!/bin/bash
# Seed 900 测试：50 runs, 2 distances, baseline_timeout=10s, 15 并行
set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/seed900_t10_logs
MASTER_LOG=$PROJ/seed900_t10_master.log

# 10 batch × 5 runs = 50 runs
SEEDS=(900 901 902 903 904 905 906 907 908 909)
RUNS=(5 5 5 5 5 5 5 5 5 5)

PROFILES=(
  seed900_t10_sr_long
  seed900_t10_sr_short
)

mkdir -p "$LOGS"
total_jobs=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
echo "$(date) START seed900 t10 test: $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee -a "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

started=0
for profile in "${PROFILES[@]}"; do
  short=$(echo "$profile" | sed 's/seed900_t10_//')
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
echo "$(date) ALL SEED900 T10 TEST DONE" | tee -a "$MASTER_LOG"
