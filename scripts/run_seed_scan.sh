#!/bin/bash
# Seed 扫描：20 seeds × 3 runs × 2 distances = 40 jobs，15 并行
set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/seed_scan_logs
MASTER_LOG=$PROJ/seed_scan_master.log

# 20 个候选 seed（覆盖广）
SEEDS=(42 100 142 200 242 300 342 400 442 500 542 600 642 700 800 900 1000 1234 2000 3000)

PROFILES=(
  seed_scan_md_kt02_sr_long
  seed_scan_md_kt02_sr_short
)

mkdir -p "$LOGS"
total_jobs=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
echo "$(date) START seed scan: $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee -a "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

started=0
for profile in "${PROFILES[@]}"; do
  short=$(echo "$profile" | sed 's/seed_scan_md_kt02_//')
  for seed in "${SEEDS[@]}"; do
    wait_for_slot
    log="$LOGS/scan_${short}_s${seed}.log"
    echo "$(date) START: $short seed=$seed -> $log" >> "$MASTER_LOG"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" --seed "$seed" --runs 3 \
      > "$log" 2>&1 &
    started=$((started + 1))
  done
done

echo "$(date) Waiting for $started jobs to finish..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL SEED SCAN DONE" | tee -a "$MASTER_LOG"
