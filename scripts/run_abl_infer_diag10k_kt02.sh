#!/bin/bash
# 消融推理 master：diag10k kt=0.2
# seeds: 42,142,...,642；每 seed 7 runs（最后一个 8），共 50 runs per variant/distance
# 并发上限 10，约占 CPU 60%+

set -euo pipefail

PROJ=/home/ubuntu/DQN9
CONDA=/home/ubuntu/miniconda3/bin/conda
ENV=ros2py310
MAX_PARALLEL=10
LOGS=$PROJ/runs/abl_diag10k_kt02_logs
MASTER_LOG=$PROJ/abl_infer_master.log

SEEDS=(42 142 242 342 442 542 642)
RUNS=(7 7 7 7 7 7 8)

PROFILES=(
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_duel_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_duel_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_mha_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_mha_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_md_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_ddqn_md_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_duel_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_duel_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_mha_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_mha_sr_short
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_md_sr_long
  ablation_20260314_diag10k_kt02_infer_cnn_dqn_md_sr_short
  ablation_20260314_diag10k_kt02_infer_mlp_sr_long
  ablation_20260314_diag10k_kt02_infer_mlp_sr_short
)

mkdir -p "$LOGS"
total_jobs=$(( ${#PROFILES[@]} * ${#SEEDS[@]} ))
echo "$(date) START infer master: $total_jobs jobs, MAX_PARALLEL=$MAX_PARALLEL" | tee -a "$MASTER_LOG"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 3
  done
}

started=0
for profile in "${PROFILES[@]}"; do
  # 从 profile 名中提取短名，用于日志文件名
  short=$(echo "$profile" | sed 's/ablation_20260314_diag10k_kt02_infer_//')
  for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    runs=${RUNS[$i]}
    wait_for_slot
    log="$LOGS/infer_${short}_s${seed}.log"
    echo "$(date) START infer: $short seed=$seed runs=$runs -> $log" >> "$MASTER_LOG"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
      python infer.py --profile "$profile" --seed "$seed" --runs "$runs" \
      > "$log" 2>&1 &
    started=$((started + 1))
  done
done

echo "$(date) Waiting for $started jobs to finish..." | tee -a "$MASTER_LOG"
wait
echo "$(date) ALL INFER DONE" | tee -a "$MASTER_LOG"
