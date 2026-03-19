#!/bin/bash
# 最终深度验证：8 个最佳 seed × 50 runs × 2 distances, 15 并行
# 每 seed 拆为 10 sub-seed × 5 runs = 50 runs
set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/final_t10_logs
MASTER_LOG=$PROJ/final_t10_master.log

# 8 个最佳 seed（扫描筛选）
BASE_SEEDS=(400 420 1690 1780 2270 2580 110 3190)

PROFILES=(final_t10_sr_long final_t10_sr_short)

mkdir -p "$LOGS"

# 每 base_seed 拆为 10 个 sub-seed (base, base+1, ..., base+9), 各 5 runs
RUNS_PER=5
SUB_COUNT=10

total_jobs=$(( ${#BASE_SEEDS[@]} * SUB_COUNT * ${#PROFILES[@]} ))
echo "$(date) START final verify: ${#BASE_SEEDS[@]} seeds x $SUB_COUNT sub x ${#PROFILES[@]} profiles = $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

started=0
for base_seed in "${BASE_SEEDS[@]}"; do
  for ((s=0; s<SUB_COUNT; s++)); do
    seed=$((base_seed + s))
    for profile in "${PROFILES[@]}"; do
      wait_for_slot
      short=$(echo "$profile" | sed 's/final_t10_//')
      log="$LOGS/${short}_s${seed}.log"
      $CONDA run --cwd "$PROJ" -n "$ENV" \
        python infer.py --profile "$profile" --seed "$seed" --runs "$RUNS_PER" \
        > "$log" 2>&1 &
      started=$((started + 1))
    done
  done
  echo "$(date) seed $base_seed dispatched ($started/$total_jobs)" >> "$MASTER_LOG"
done

echo "$(date) All $started jobs started, waiting..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL FINAL VERIFY DONE ($started jobs)" | tee -a "$MASTER_LOG"
