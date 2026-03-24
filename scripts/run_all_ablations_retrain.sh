#!/usr/bin/env bash
# ============================================================
# 全部消融实验重训（goal_tolerance 1.0m → 0.3m）
# 分两批执行，避免 OOM：
#   批 1: 架构消融(9) + 标量观测(1) = 10 并行 (~20GB VRAM)
#   批 2: AM×DQfD(2) + 分辨率(3)   = 5 并行  (~10GB VRAM)
# Usage: bash scripts/run_all_ablations_retrain.sh
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/retrain_20260324_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

# ============================================================
# 批 1 训练 profiles (10 个)
# ============================================================
BATCH1_TRAIN=(
    ablation_20260314_diag10k_kt02_cnn_ddqn
    ablation_20260314_diag10k_kt02_cnn_ddqn_duel
    ablation_20260314_diag10k_kt02_cnn_ddqn_mha
    ablation_20260314_diag10k_kt02_cnn_ddqn_md
    ablation_20260314_diag10k_kt02_cnn_dqn
    ablation_20260314_diag10k_kt02_cnn_dqn_duel
    ablation_20260314_diag10k_kt02_cnn_dqn_mha
    ablation_20260314_diag10k_kt02_cnn_dqn_md
    ablation_20260314_diag10k_kt02_mlp
    ablation_20260323_scalar_only
)

# 批 1 推理 profiles (20 个)
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
# 批 2 训练 profiles (5 个)
# ============================================================
BATCH2_TRAIN=(
    ablation_20260315_amdqfd_noAM
    ablation_20260315_amdqfd_noDQfD
    ablation_20260318_resolution_n8
    ablation_20260318_resolution_n16
    ablation_20260318_resolution_n24
)

# 批 2 推理 profiles — 分辨率 (6 个，直接 profile 调用)
BATCH2_INFER_RESOLUTION=(
    ablation_20260318_resolution_n8_sr_long
    ablation_20260318_resolution_n8_sr_short
    ablation_20260318_resolution_n16_sr_long
    ablation_20260318_resolution_n16_sr_short
    ablation_20260318_resolution_n24_sr_long
    ablation_20260318_resolution_n24_sr_short
)

# 批 2 推理 — AM×DQfD (多 seed，复用 run_abl_amdqfd.sh 的逻辑)
BATCH2_INFER_AMDQFD=(
    ablation_20260315_amdqfd_infer_full_sr_long
    ablation_20260315_amdqfd_infer_full_sr_short
    ablation_20260315_amdqfd_infer_noDQfD_sr_long
    ablation_20260315_amdqfd_infer_noDQfD_sr_short
    ablation_20260315_amdqfd_infer_noAM_sr_long
    ablation_20260315_amdqfd_infer_noAM_sr_short
)
AMDQFD_SEEDS=(100 200 300 400 500 600 700)
AMDQFD_RUNS=(7 7 7 7 7 7 8)
MAX_PARALLEL=16

# ============================================================
# 辅助函数
# ============================================================
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 3
    done
}

run_phase_train() {
    local phase_name="$1"
    shift
    local profiles=("$@")
    local n=${#profiles[@]}

    echo "$(date) ===== $phase_name TRAINING ($n parallel) ====="
    PIDS=()
    for p in "${profiles[@]}"; do
        short="${p##*_kt02_}"
        [ "$short" = "$p" ] && short="${p##*_}"
        log="$LOGDIR/train_${short}_${TS}.log"
        echo "$(date) START train: $p -> $log"
        nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
            python train.py --profile "$p" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done

    echo "$(date) Waiting for $n training jobs: ${PIDS[*]}"
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "$(date) FAIL: PID $pid"
            FAIL=$((FAIL+1))
        fi
    done

    if [ $FAIL -gt 0 ]; then
        echo "$(date) WARNING: $FAIL/$n training job(s) failed in $phase_name"
    else
        echo "$(date) $phase_name: All $n training jobs completed successfully."
    fi
}

run_phase_infer_simple() {
    local phase_name="$1"
    shift
    local profiles=("$@")
    local n=${#profiles[@]}

    echo "$(date) ===== $phase_name INFERENCE ($n parallel) ====="
    PIDS=()
    for p in "${profiles[@]}"; do
        short="${p##*_kt02_}"
        [ "$short" = "$p" ] && short="${p##*_}"
        log="$LOGDIR/infer_${short}_${TS}.log"
        echo "$(date) START infer: $p -> $log"
        nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
            python infer.py --profile "$p" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done

    echo "$(date) Waiting for $n inference jobs: ${PIDS[*]}"
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "$(date) FAIL: PID $pid"
            FAIL=$((FAIL+1))
        fi
    done

    if [ $FAIL -gt 0 ]; then
        echo "$(date) WARNING: $FAIL/$n inference job(s) failed in $phase_name"
    else
        echo "$(date) $phase_name: All $n inference jobs completed successfully."
    fi
}

run_amdqfd_infer() {
    echo "$(date) ===== BATCH2 AM×DQfD INFERENCE (multi-seed) ====="
    started=0
    for profile in "${BATCH2_INFER_AMDQFD[@]}"; do
        short=$(echo "$profile" | sed 's/ablation_20260315_amdqfd_infer_//')
        for i in "${!AMDQFD_SEEDS[@]}"; do
            seed=${AMDQFD_SEEDS[$i]}
            runs=${AMDQFD_RUNS[$i]}
            wait_for_slot
            log="$LOGDIR/infer_amdqfd_${short}_s${seed}_${TS}.log"
            echo "$(date) START infer: $short seed=$seed runs=$runs -> $log"
            nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
                python infer.py --profile "$profile" --seed "$seed" --runs "$runs" \
                > "$log" 2>&1 &
            started=$((started + 1))
        done
    done

    echo "$(date) Waiting for $started AM×DQfD inference jobs..."
    wait
    echo "$(date) AM×DQfD INFER DONE ($started jobs)"
}

# ============================================================
# 主流程
# ============================================================
echo "$(date) ======================================================"
echo "$(date) 全部消融实验重训开始 (goal_tolerance 0.3m)"
echo "$(date) ======================================================"

# ------ 批 1: 架构消融 + 标量观测 ------
run_phase_train "BATCH1" "${BATCH1_TRAIN[@]}"
run_phase_infer_simple "BATCH1" "${BATCH1_INFER[@]}"

# ------ 批 2: AM×DQfD + 分辨率 ------
run_phase_train "BATCH2" "${BATCH2_TRAIN[@]}"
run_phase_infer_simple "BATCH2-RESOLUTION" "${BATCH2_INFER_RESOLUTION[@]}"
run_amdqfd_infer

echo "$(date) ======================================================"
echo "$(date) 全部消融实验重训完成"
echo "$(date) Logs: $LOGDIR"
echo "$(date) ======================================================"
