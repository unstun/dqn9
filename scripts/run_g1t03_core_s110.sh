#!/usr/bin/env bash
# ============================================================
# g1t03 核心对比: MD-DDQN(V1) vs baselines, seed=110
# Train 1.0m -> Infer 0.3m, 50 runs per suite
# Usage: bash scripts/run_g1t03_core_s110.sh
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN9"
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
LOGDIR="$PROJ/runs/g1t03_core_s110_$(date +%Y%m%d)_logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)

PROFILES=(
    repro_20260325_g1t03_core_s110_sr_long
    repro_20260325_g1t03_core_s110_sr_short
)

PIDS=()
for p in "${PROFILES[@]}"; do
    log="$LOGDIR/${p}_${TS}.log"
    echo "$(date) START: $p"
    nohup "$CONDA" run --cwd "$PROJ" -n "$ENV" \
        python infer.py --profile "$p" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo "$(date) Waiting for ${#PIDS[@]} jobs: ${PIDS[*]}"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "$(date) FAIL: PID $pid"; FAIL=$((FAIL+1))
    fi
done

[ $FAIL -gt 0 ] && echo "$(date) WARNING: $FAIL/${#PIDS[@]} failed" \
                 || echo "$(date) All ${#PIDS[@]} OK"
echo "$(date) Logs: $LOGDIR"
