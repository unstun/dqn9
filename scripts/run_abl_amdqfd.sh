#!/usr/bin/env bash
# AM x DQfD 组件消融: 2 个训练并行 + 6 个推理（多 seed）
# Usage: bash scripts/run_abl_amdqfd.sh [train|infer|all]
# 远程 4090 服务器执行，2 训练并行 (~2-3GB VRAM each)
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/abl_amdqfd_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

TRAIN_PROFILES=(
    ablation_20260315_amdqfd_noDQfD
    ablation_20260315_amdqfd_noAM
)

INFER_PROFILES=(
    ablation_20260315_amdqfd_infer_full_sr_long
    ablation_20260315_amdqfd_infer_full_sr_short
    ablation_20260315_amdqfd_infer_noDQfD_sr_long
    ablation_20260315_amdqfd_infer_noDQfD_sr_short
    ablation_20260315_amdqfd_infer_noAM_sr_long
    ablation_20260315_amdqfd_infer_noAM_sr_short
)

# 推理多 seed: 100,200,...,700; 每 seed 7 runs (最后一个 8)，共 50 runs
SEEDS=(100 200 300 400 500 600 700)
RUNS=(7 7 7 7 7 7 8)
MAX_PARALLEL=10

run_train() {
    echo "$(date) ===== TRAINING PHASE (2 parallel) ====="
    PIDS=()
    for p in "${TRAIN_PROFILES[@]}"; do
        short="${p#ablation_20260315_amdqfd_}"
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
        echo "$(date) All 2 training jobs completed successfully."
    fi
}

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 3
    done
}

run_infer() {
    echo "$(date) ===== INFERENCE PHASE (multi-seed) ====="
    started=0
    for profile in "${INFER_PROFILES[@]}"; do
        short=$(echo "$profile" | sed 's/ablation_20260315_amdqfd_infer_//')
        for i in "${!SEEDS[@]}"; do
            seed=${SEEDS[$i]}
            runs=${RUNS[$i]}
            wait_for_slot
            log="$LOGDIR/infer_${short}_s${seed}_${TS}.log"
            echo "$(date) START infer: $short seed=$seed runs=$runs -> $log"
            nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
                python infer.py --profile "$profile" --seed "$seed" --runs "$runs" \
                > "$log" 2>&1 &
            started=$((started + 1))
        done
    done

    echo "$(date) Waiting for $started inference jobs..."
    wait
    echo "$(date) ALL INFER DONE ($started jobs)"
}

MODE="${1:-all}"
case "$MODE" in
    train) run_train ;;
    infer) run_infer ;;
    all)   run_train && run_infer ;;
    *)     echo "Usage: $0 [train|infer|all]"; exit 1 ;;
esac

echo "$(date) DONE. Logs: $LOGDIR"
