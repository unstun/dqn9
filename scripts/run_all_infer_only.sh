#!/usr/bin/env bash
# ============================================================
# 纯推理脚本（0.3m 重训模型已就绪，仅跑推理）
# 复用 run_all_ablations_retrain.sh 的推理框架
# Usage: bash scripts/run_all_infer_only.sh
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/reinfer_$(date +%Y%m%d)_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

MAX_PARALLEL=16

# ============================================================
# 批 1: 架构消融推理 (20 个, 单 seed, 50 runs)
# ============================================================
BATCH1_INFER=(
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
    ablation_20260323_scalar_only_sr_long
    ablation_20260323_scalar_only_sr_short
)

# ============================================================
# 批 2a: 分辨率推理 (6 个, 单 seed)
# ============================================================
BATCH2_RESOLUTION=(
    ablation_20260318_resolution_n8_sr_long
    ablation_20260318_resolution_n8_sr_short
    ablation_20260318_resolution_n16_sr_long
    ablation_20260318_resolution_n16_sr_short
    ablation_20260318_resolution_n24_sr_long
    ablation_20260318_resolution_n24_sr_short
)

# ============================================================
# 批 2b: AM×DQfD 推理 (6 profiles × 7 seeds = 42 推理)
# ============================================================
BATCH2_AMDQFD=(
    ablation_20260315_amdqfd_infer_full_sr_long
    ablation_20260315_amdqfd_infer_full_sr_short
    ablation_20260315_amdqfd_infer_noDQfD_sr_long
    ablation_20260315_amdqfd_infer_noDQfD_sr_short
    ablation_20260315_amdqfd_infer_noAM_sr_long
    ablation_20260315_amdqfd_infer_noAM_sr_short
)
AMDQFD_SEEDS=(100 200 300 400 500 600 700)
AMDQFD_RUNS=(7 7 7 7 7 7 8)

# ============================================================
# 辅助函数
# ============================================================
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 3
    done
}

run_simple_infer() {
    local phase="$1"; shift
    local profiles=("$@")
    local n=${#profiles[@]}
    echo "$(date) ===== $phase INFERENCE ($n profiles) ====="
    local PIDS=()
    for p in "${profiles[@]}"; do
        short="${p##*_kt02_}"
        [ "$short" = "$p" ] && short="${p##*_}"
        log="$LOGDIR/infer_${short}_${TS}.log"
        echo "$(date) START: $p"
        wait_for_slot
        nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
            python infer.py --profile "$p" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done
    echo "$(date) Waiting for $n jobs: ${PIDS[*]}"
    local FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "$(date) FAIL: PID $pid"; FAIL=$((FAIL+1))
        fi
    done
    [ $FAIL -gt 0 ] && echo "$(date) WARNING: $FAIL/$n failed in $phase" \
                     || echo "$(date) $phase: All $n OK"
}

run_amdqfd_infer() {
    echo "$(date) ===== AM×DQfD INFERENCE (multi-seed) ====="
    local started=0
    for profile in "${BATCH2_AMDQFD[@]}"; do
        short=$(echo "$profile" | sed 's/ablation_20260315_amdqfd_infer_//')
        for i in "${!AMDQFD_SEEDS[@]}"; do
            seed=${AMDQFD_SEEDS[$i]}
            runs=${AMDQFD_RUNS[$i]}
            wait_for_slot
            log="$LOGDIR/infer_amdqfd_${short}_s${seed}_${TS}.log"
            echo "$(date) START: $short seed=$seed runs=$runs"
            nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
                python infer.py --profile "$profile" --seed "$seed" --runs "$runs" \
                > "$log" 2>&1 &
            started=$((started + 1))
        done
    done
    echo "$(date) Waiting for $started AM×DQfD jobs..."
    wait
    echo "$(date) AM×DQfD DONE ($started jobs)"
}

# ============================================================
# 主流程
# ============================================================
echo "$(date) ======================================================"
echo "$(date) 0.3m 重训模型 — 纯推理开始"
echo "$(date) ======================================================"

run_simple_infer "BATCH1-ARCH" "${BATCH1_INFER[@]}"
run_simple_infer "BATCH2-RESOLUTION" "${BATCH2_RESOLUTION[@]}"
run_amdqfd_infer

echo "$(date) ======================================================"
echo "$(date) 全部推理完成"
echo "$(date) Logs: $LOGDIR"
echo "$(date) ======================================================"
