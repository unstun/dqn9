#!/usr/bin/env bash
# Ablation: 9 architecture variants, diag 10k episodes, kt=0.2
# Usage: bash scripts/run_abl_diag10k_kt02.sh [train|infer|all]
# Designed for remote 4090 server (single GPU, sequential within each variant)
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/abl_diag10k_kt02_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

TRAIN_PROFILES=(
    ablation_20260314_diag10k_kt02_cnn_ddqn
    ablation_20260314_diag10k_kt02_cnn_ddqn_duel
    ablation_20260314_diag10k_kt02_cnn_ddqn_mha
    ablation_20260314_diag10k_kt02_cnn_ddqn_md
    ablation_20260314_diag10k_kt02_cnn_dqn
    ablation_20260314_diag10k_kt02_cnn_dqn_duel
    ablation_20260314_diag10k_kt02_cnn_dqn_mha
    ablation_20260314_diag10k_kt02_cnn_dqn_md
    ablation_20260314_diag10k_kt02_mlp
)

INFER_PROFILES=(
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

run_train() {
    echo "$(date) ===== TRAINING PHASE (9 parallel) ====="
    PIDS=()
    for p in "${TRAIN_PROFILES[@]}"; do
        short="${p#ablation_20260314_diag10k_kt02_}"
        log="$LOGDIR/train_${short}_${TS}.log"
        echo "$(date) START train: $short -> $log"
        nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
            python train.py --profile "$p" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done

    echo "$(date) Waiting for ${#PIDS[@]} training jobs: ${PIDS[*]}"
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "$(date) FAIL: PID $pid"
            FAIL=$((FAIL+1))
        fi
    done

    if [ $FAIL -gt 0 ]; then
        echo "$(date) WARNING: $FAIL training job(s) failed. Check logs in $LOGDIR"
    else
        echo "$(date) All 9 training jobs completed successfully."
    fi
}

run_infer() {
    echo "$(date) ===== INFERENCE PHASE (18 parallel) ====="
    PIDS=()
    for p in "${INFER_PROFILES[@]}"; do
        short="${p#ablation_20260314_diag10k_kt02_}"
        log="$LOGDIR/infer_${short}_${TS}.log"
        echo "$(date) START infer: $short -> $log"
        nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
            python infer.py --profile "$p" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done

    echo "$(date) Waiting for ${#PIDS[@]} inference jobs: ${PIDS[*]}"
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "$(date) FAIL: PID $pid"
            FAIL=$((FAIL+1))
        fi
    done

    if [ $FAIL -gt 0 ]; then
        echo "$(date) WARNING: $FAIL inference job(s) failed. Check logs in $LOGDIR"
    else
        echo "$(date) All 18 inference jobs completed successfully."
    fi
}

MODE="${1:-all}"
case "$MODE" in
    train) run_train ;;
    infer) run_infer ;;
    all)   run_train && run_infer ;;
    *)     echo "Usage: $0 [train|infer|all]"; exit 1 ;;
esac

echo "$(date) DONE. Logs: $LOGDIR"
