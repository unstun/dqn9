#!/usr/bin/env bash
# ============================================================
# g1t03 消融统一推理: seed=110, 50 runs, goal_tolerance=0.3m
# Train 1.0m (V1) -> Infer 0.3m, 全部消融实验统一 seed
# 配置: configs/ablation_20260326_g1t03_s110_*.json
# Usage: bash scripts/run_g1t03_abl_s110.sh
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/g1t03_abl_s110_$(date +%Y%m%d)_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

MAX_PARALLEL=16

# ============================================================
# 全部 32 个推理 profile (seed=110, 50 runs)
# ============================================================
PROFILES=(
    # --- 架构消融: 8 CNN + MLP + Scalar-Only (20 profiles) ---
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_duel_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_duel_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_mha_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_mha_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_md_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_ddqn_md_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_dqn_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_dqn_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_dqn_duel_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_dqn_duel_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_dqn_mha_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_dqn_mha_sr_short
    ablation_20260326_g1t03_s110_infer_cnn_dqn_md_sr_long
    ablation_20260326_g1t03_s110_infer_cnn_dqn_md_sr_short
    ablation_20260326_g1t03_s110_infer_mlp_sr_long
    ablation_20260326_g1t03_s110_infer_mlp_sr_short
    ablation_20260326_g1t03_s110_scalar_only_sr_long
    ablation_20260326_g1t03_s110_scalar_only_sr_short
    # --- 分辨率消融 (6 profiles) ---
    ablation_20260326_g1t03_s110_resolution_n8_sr_long
    ablation_20260326_g1t03_s110_resolution_n8_sr_short
    ablation_20260326_g1t03_s110_resolution_n16_sr_long
    ablation_20260326_g1t03_s110_resolution_n16_sr_short
    ablation_20260326_g1t03_s110_resolution_n24_sr_long
    ablation_20260326_g1t03_s110_resolution_n24_sr_short
    # --- AM x DQfD 消融 (6 profiles) ---
    ablation_20260326_g1t03_s110_amdqfd_infer_full_sr_long
    ablation_20260326_g1t03_s110_amdqfd_infer_full_sr_short
    ablation_20260326_g1t03_s110_amdqfd_infer_noDQfD_sr_long
    ablation_20260326_g1t03_s110_amdqfd_infer_noDQfD_sr_short
    ablation_20260326_g1t03_s110_amdqfd_infer_noAM_sr_long
    ablation_20260326_g1t03_s110_amdqfd_infer_noAM_sr_short
)

# ============================================================
# 辅助函数
# ============================================================
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 3
    done
}

# ============================================================
# 主流程: 统一并行推理
# ============================================================
N=${#PROFILES[@]}
echo "$(date) ======================================================"
echo "$(date) g1t03 Ablation Unified Inference: seed=110, 50 runs"
echo "$(date) Total profiles: $N, MAX_PARALLEL=$MAX_PARALLEL"
echo "$(date) ======================================================"

PIDS=()
for p in "${PROFILES[@]}"; do
    short="${p##*g1t03_s110_}"
    log="$LOGDIR/infer_${short}_${TS}.log"
    echo "$(date) START: $short"
    wait_for_slot
    nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
        python infer.py --profile "$p" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo "$(date) Waiting for $N jobs: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "$(date) FAIL: PID $pid"; FAIL=$((FAIL+1))
    fi
done

[ $FAIL -gt 0 ] && echo "$(date) WARNING: $FAIL/$N failed" \
                 || echo "$(date) All $N OK"
echo "$(date) Logs: $LOGDIR"
echo "$(date) ======================================================"
echo "$(date) g1t03 s110 ablation DONE"
echo "$(date) ======================================================"
