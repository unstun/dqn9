#!/usr/bin/env bash
# Reward ablation: 18 parallel trains → 36 parallel infers
# Run on remote GPU server: bash scripts/run_reward_ablation_18.sh
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOG_DIR="$PROJ/runs/reward_ablation_logs"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
    T1a_kt02 T1b_kt04 T1c_kt08
    T2a_smooth_mild T2b_smooth_mid T2c_smooth_aggr
    T3a_clip8 T3b_clip12 T3c_clip20
    T4a_obs_mild T4b_obs_mid T4c_obs_aggr
    T5a_eff03 T5b_eff08 T5c_eff15
    C1_time_smooth_clip C2_plus_obs C3_full
)

echo "=== Phase 1: Launching ${#EXPERIMENTS[@]} parallel training runs ==="
TRAIN_PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
    profile="reward_abl_${exp}"
    log="$LOG_DIR/train_${exp}.log"
    echo "  Starting train: $profile → $log"
    $CONDA run --cwd "$PROJ" -n "$ENV" \
        python train.py --profile "$profile" \
        > "$log" 2>&1 &
    TRAIN_PIDS+=($!)
done

echo "  ${#TRAIN_PIDS[@]} training processes launched. PIDs: ${TRAIN_PIDS[*]}"
echo "  Waiting for all training to complete..."

# Wait for all training processes, track failures
TRAIN_FAIL=0
for i in "${!TRAIN_PIDS[@]}"; do
    pid=${TRAIN_PIDS[$i]}
    exp=${EXPERIMENTS[$i]}
    if wait "$pid"; then
        echo "  ✓ train $exp done (PID $pid)"
    else
        echo "  ✗ train $exp FAILED (PID $pid, exit=$?)"
        TRAIN_FAIL=$((TRAIN_FAIL + 1))
    fi
done

echo "=== Training complete: $((${#EXPERIMENTS[@]} - TRAIN_FAIL))/${#EXPERIMENTS[@]} succeeded ==="

if [ "$TRAIN_FAIL" -gt 0 ]; then
    echo "WARNING: $TRAIN_FAIL training runs failed. Continuing with inference for successful ones."
fi

echo ""
echo "=== Phase 2: Launching inference (2 modes × ${#EXPERIMENTS[@]} = $((${#EXPERIMENTS[@]} * 2))) ==="
INFER_PIDS=()
INFER_NAMES=()
for exp in "${EXPERIMENTS[@]}"; do
    train_out="reward_abl_${exp}"
    # Check if training produced models
    train_dir=$(ls -dt "$PROJ/runs/${train_out}/train_"* 2>/dev/null | head -1)
    if [ -z "$train_dir" ] || [ ! -d "$train_dir/models" ]; then
        echo "  SKIP $exp: no training output found"
        continue
    fi

    for mode in sr_long sr_short; do
        profile="reward_abl_infer_${exp}_${mode}"
        log="$LOG_DIR/infer_${exp}_${mode}.log"
        echo "  Starting infer: $profile → $log"
        $CONDA run --cwd "$PROJ" -n "$ENV" \
            python infer.py --profile "$profile" \
            > "$log" 2>&1 &
        INFER_PIDS+=($!)
        INFER_NAMES+=("${exp}_${mode}")
    done
done

echo "  ${#INFER_PIDS[@]} inference processes launched."
echo "  Waiting for all inference to complete..."

INFER_FAIL=0
for i in "${!INFER_PIDS[@]}"; do
    pid=${INFER_PIDS[$i]}
    name=${INFER_NAMES[$i]}
    if wait "$pid"; then
        echo "  ✓ infer $name done (PID $pid)"
    else
        echo "  ✗ infer $name FAILED (PID $pid, exit=$?)"
        INFER_FAIL=$((INFER_FAIL + 1))
    fi
done

echo ""
echo "==========================================="
echo "  REWARD ABLATION COMPLETE"
echo "  Training: $((${#EXPERIMENTS[@]} - TRAIN_FAIL))/${#EXPERIMENTS[@]} OK"
echo "  Inference: $((${#INFER_PIDS[@]} - INFER_FAIL))/${#INFER_PIDS[@]} OK"
echo "  Logs: $LOG_DIR/"
echo "==========================================="
