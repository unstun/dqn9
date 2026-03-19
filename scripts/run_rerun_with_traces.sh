#!/bin/bash
# 三组实验全量重跑（带 save_traces），输出 .1 后缀
# Usage: bash scripts/run_rerun_with_traces.sh [exp1|exp2|exp3|all]
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/rerun_traces_logs"
MASTER_LOG="$LOGDIR/master.log"
MAX_PARALLEL=10

mkdir -p "$LOGDIR"
log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$MASTER_LOG"; }

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 2
  done
}

# ============================================================
# EXP1: 核心对比 CNN-DDQN+MD vs RRT* vs LO-HA*
#   seed 420, 拆为 10 sub-seed (420-429) x 5 runs = 50 runs
# ============================================================
run_exp1() {
  log "===== EXP1: Core comparison (seed 420, 50 runs) ====="
  local profiles=(final_t10_sr_long final_t10_sr_short)
  local base_seed=420
  local sub_count=10
  local runs_per=5

  for profile in "${profiles[@]}"; do
    out_name="${profile}.1"
    for ((s=0; s<sub_count; s++)); do
      seed=$((base_seed + s))
      wait_for_slot
      logf="$LOGDIR/exp1_${profile##final_t10_}_s${seed}.log"
      log "  exp1: $profile seed=$seed runs=$runs_per -> $out_name"
      $CONDA run --cwd "$PROJ" -n "$ENV" \
        python infer.py --profile "$profile" \
          --seed "$seed" --runs "$runs_per" \
          --save-traces --forest-baseline-save-traces \
          --out "$out_name" \
        > "$logf" 2>&1 &
    done
  done
  log "EXP1 dispatched, waiting..."
  wait
  log "EXP1 DONE"
}

# ============================================================
# EXP2: 模块消融 10 变体
#   seeds 42/142/242/342/442/542/642, 拆 7+7+7+7+7+7+8 = 50 runs
# ============================================================
run_exp2() {
  log "===== EXP2: Module ablation (10 variants x 7 seeds) ====="
  local profiles=(
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
  local seeds=(42 142 242 342 442 542 642)
  local runs=(7 7 7 7 7 7 8)

  for profile in "${profiles[@]}"; do
    # 从 profile 中提取 out 名并加 .1
    local base_out
    base_out=$(python3 -c "
import json, sys
with open('$PROJ/configs/${profile}.json') as f:
    print(json.load(f)['infer']['out'])
")
    local out_name="${base_out}.1"

    for i in "${!seeds[@]}"; do
      seed=${seeds[$i]}
      r=${runs[$i]}
      wait_for_slot
      short=$(echo "$profile" | sed 's/ablation_20260314_diag10k_kt02_infer_//')
      logf="$LOGDIR/exp2_${short}_s${seed}.log"
      log "  exp2: $short seed=$seed runs=$r -> $out_name"
      $CONDA run --cwd "$PROJ" -n "$ENV" \
        python infer.py --profile "$profile" \
          --seed "$seed" --runs "$r" \
          --save-traces \
          --out "$out_name" \
        > "$logf" 2>&1 &
    done
  done
  log "EXP2 dispatched, waiting..."
  wait
  log "EXP2 DONE"
}

# ============================================================
# EXP3: AM x DQfD 组件消融 3 变体
#   seeds 100/200/300/400/500/600/700, 拆 7+7+7+7+7+7+8 = 50 runs
# ============================================================
run_exp3() {
  log "===== EXP3: AM x DQfD ablation (3 variants x 7 seeds) ====="
  local profiles=(
    ablation_20260315_amdqfd_infer_full_sr_long
    ablation_20260315_amdqfd_infer_full_sr_short
    ablation_20260315_amdqfd_infer_noDQfD_sr_long
    ablation_20260315_amdqfd_infer_noDQfD_sr_short
    ablation_20260315_amdqfd_infer_noAM_sr_long
    ablation_20260315_amdqfd_infer_noAM_sr_short
  )
  local seeds=(100 200 300 400 500 600 700)
  local runs=(7 7 7 7 7 7 8)

  for profile in "${profiles[@]}"; do
    local base_out
    base_out=$(python3 -c "
import json, sys
with open('$PROJ/configs/${profile}.json') as f:
    print(json.load(f)['infer']['out'])
")
    local out_name="${base_out}.1"

    for i in "${!seeds[@]}"; do
      seed=${seeds[$i]}
      r=${runs[$i]}
      wait_for_slot
      short=$(echo "$profile" | sed 's/ablation_20260315_amdqfd_infer_//')
      logf="$LOGDIR/exp3_${short}_s${seed}.log"
      log "  exp3: $short seed=$seed runs=$r -> $out_name"
      $CONDA run --cwd "$PROJ" -n "$ENV" \
        python infer.py --profile "$profile" \
          --seed "$seed" --runs "$r" \
          --save-traces --forest-baseline-save-traces \
          --out "$out_name" \
        > "$logf" 2>&1 &
    done
  done
  log "EXP3 dispatched, waiting..."
  wait
  log "EXP3 DONE"
}

# ============================================================
MODE="${1:-all}"
case "$MODE" in
  exp1) run_exp1 ;;
  exp2) run_exp2 ;;
  exp3) run_exp3 ;;
  all)  run_exp1 && run_exp2 && run_exp3 ;;
  *)    echo "Usage: $0 [exp1|exp2|exp3|all]"; exit 1 ;;
esac

log "ALL DONE. Logs: $LOGDIR"
