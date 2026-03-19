#!/bin/bash
# Seed 2930 验证：50 runs, 2 distances, baseline_timeout=10s, 15 并行
set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/seed2930_logs
MASTER_LOG=$PROJ/seed2930_master.log

# 10 sub-seed × 5 runs = 50 runs
SEEDS=(2930 2931 2932 2933 2934 2935 2936 2937 2938 2939)

PROFILES=(final_t10_sr_long final_t10_sr_short)

mkdir -p "$LOGS"
total_jobs=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
echo "$(date) START seed2930 verify: $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

started=0
for seed in "${SEEDS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    wait_for_slot
    short=$(echo "$profile" | sed 's/final_t10_//')
    log="$LOGS/${short}_s${seed}.log"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" --seed "$seed" --runs 5 \
      > "$log" 2>&1 &
    started=$((started + 1))
  done
done

echo "$(date) All $started jobs started, waiting..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL SEED2930 VERIFY DONE ($started jobs)" | tee -a "$MASTER_LOG"
