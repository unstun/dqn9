#!/bin/bash
# 核心对比推理（无 MPC）：CNN-DDQN+MD vs RRT* vs LO-HA*
# seeds: 42,142,...,642；每 seed 7 runs（最后一个 8），共 50 runs per distance
# 3 算法 × 5 并发 = 15 并行

set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=15
LOGS=$PROJ/runs/mpc_md_kt02_logs
MASTER_LOG=$PROJ/mpc_md_infer_master.log

SEEDS=(42 142 242 342 442 542 642)
RUNS=(7 7 7 7 7 7 8)

PROFILES=(
  mpc_md_kt02_sr_long
  mpc_md_kt02_sr_short
)

mkdir -p "$LOGS"
total_jobs=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
echo "$(date) START core comparison infer (no MPC): $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee -a "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 3
  done
}

started=0
for profile in "${PROFILES[@]}"; do
  short=$(echo "$profile" | sed 's/mpc_md_kt02_//')
  for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    runs=${RUNS[$i]}
    wait_for_slot
    log="$LOGS/infer_${short}_s${seed}.log"
    echo "$(date) START: $short seed=$seed runs=$runs -> $log" >> "$MASTER_LOG"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" --seed "$seed" --runs "$runs" \
      > "$log" 2>&1 &
    started=$((started + 1))
  done
done

echo "$(date) Waiting for $started jobs to finish..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL CORE COMPARISON INFER DONE" | tee -a "$MASTER_LOG"
