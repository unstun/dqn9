#!/bin/bash
# 大规模 seed 扫描：500 seeds (0-4990 step 10), 5 runs each, 2 distances, 15 并行
# baseline_timeout=10s, EDT diag collision
# 预计运行时间 ~45-60 分钟
set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/scan_t10_logs
MASTER_LOG=$PROJ/scan_t10_master.log

PROFILES=(scan_t10_sr_long scan_t10_sr_short)

# 500 seeds: 0, 10, 20, ..., 4990
SEED_START=0
SEED_END=4990
SEED_STEP=10
RUNS_PER=5

mkdir -p "$LOGS"

total_seeds=$(( (SEED_END - SEED_START) / SEED_STEP + 1 ))
total_jobs=$(( total_seeds * ${#PROFILES[@]} ))
echo "$(date) START large scan: $total_seeds seeds x ${#PROFILES[@]} profiles = $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

started=0
for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do
  for profile in "${PROFILES[@]}"; do
    wait_for_slot
    short=$(echo "$profile" | sed 's/scan_t10_//')
    log="$LOGS/${short}_s${seed}.log"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" --seed "$seed" --runs "$RUNS_PER" \
      > "$log" 2>&1 &
    started=$((started + 1))
    # 每100个job打一次日志
    if (( started % 100 == 0 )); then
      echo "$(date) PROGRESS: $started/$total_jobs jobs started" >> "$MASTER_LOG"
    fi
  done
done

echo "$(date) All $started jobs started, waiting..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL LARGE SCAN DONE ($started jobs)" | tee -a "$MASTER_LOG"
