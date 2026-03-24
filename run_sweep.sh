#!/bin/bash
# 64 个推理任务，每批 16 并行（匹配 16 核 CPU），分 4 批执行
CONDA="$HOME/miniconda3/bin/conda"
ENV="ros2py310"
PROJ="$HOME/DQN9"
LOGDIR="$PROJ/runs/sweep_logs"
mkdir -p "$LOGDIR"

BATCH=16
total=0
batch_count=0

for seed in $(seq 100 131); do
    for suite in long short; do
        profile="sweep/sweep_s${seed}_${suite}"
        log="$LOGDIR/s${seed}_${suite}.log"
        $CONDA run --cwd "$PROJ" -n "$ENV" python infer.py --profile "$profile" > "$log" 2>&1 &
        total=$((total + 1))
        batch_count=$((batch_count + 1))

        if [ $batch_count -ge $BATCH ]; then
            echo "[$(date)] 等待第 $((total / BATCH)) 批 ($BATCH 个) 完成..."
            wait
            batch_count=0
            echo "[$(date)] 批次完成，已完成 $total/64"
        fi
    done
done

# 等剩余
if [ $batch_count -gt 0 ]; then
    echo "[$(date)] 等待最后 $batch_count 个完成..."
    wait
fi

echo "[$(date)] 全部 $total 个任务已完成！"
