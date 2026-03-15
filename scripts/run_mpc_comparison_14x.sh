#!/usr/bin/env bash
# MPC comparison: kt02 DRL+MPC vs LO-HA*+MPC vs SS-RRT*+MPC
# 2 modes × 7 chunks = 14 parallel processes
# Each chunk: 7 runs (last chunk: 8 runs to reach 50 total)
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOG_DIR="$PROJ/runs/mpc_comparison_logs"
mkdir -p "$LOG_DIR"

TOTAL_RUNS=50
CHUNKS=7
RUNS_PER_CHUNK=7
LAST_CHUNK_RUNS=8   # 7*6 + 8 = 50

PROFILES=("reward_abl_mpc_kt02_sr_long" "reward_abl_mpc_kt02_sr_short")

echo "=== Launching MPC comparison: ${#PROFILES[@]} profiles × $CHUNKS chunks = $(( ${#PROFILES[@]} * CHUNKS )) parallel ==="

PIDS=()
NAMES=()

for profile in "${PROFILES[@]}"; do
    for chunk_idx in $(seq 0 $((CHUNKS - 1))); do
        if [ "$chunk_idx" -eq $((CHUNKS - 1)) ]; then
            runs=$LAST_CHUNK_RUNS
        else
            runs=$RUNS_PER_CHUNK
        fi

        chunk_out="${profile}_chunk${chunk_idx}"
        log="$LOG_DIR/${profile}_chunk${chunk_idx}.log"

        echo "  Starting: $chunk_out ($runs runs) → $log"
        $CONDA run --cwd "$PROJ" -n "$ENV" \
            python infer.py --profile "$profile" \
            --runs "$runs" \
            --seed "$((42 + chunk_idx * 100))" \
            --out "$chunk_out" \
            > "$log" 2>&1 &
        PIDS+=($!)
        NAMES+=("$chunk_out")
    done
done

echo ""
echo "  ${#PIDS[@]} processes launched. PIDs: ${PIDS[*]}"
echo "  Waiting for all to complete..."

FAIL=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    if wait "$pid"; then
        echo "  ✓ $name done (PID $pid)"
    else
        echo "  ✗ $name FAILED (PID $pid, exit=$?)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Complete: $(( ${#PIDS[@]} - FAIL ))/${#PIDS[@]} succeeded, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    echo "Check logs in $LOG_DIR for failures."
fi
