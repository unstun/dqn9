#!/bin/bash
# ============================================================
# §3.5 DM-DQN 组件消融：训练完成后自动推理
# 等待 w/o AM + w/o DQfD 训练结束 → 启动 6 轮推理
# 用法: nohup bash scripts/auto_infer_35_dmdqn.sh > runs/auto_infer_35.log 2>&1 &
# ============================================================

PROJ=~/DQN9
CONDA=~/miniconda3/bin/conda
ENV=ros2py310

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] === §3.5 auto-inference script started ==="

# ---- Phase 1: 等待训练完成 ----
echo "[$(ts)] Waiting for training to finish..."

while true; do
    # 检查训练进程是否还在
    N_TRAIN=$(ps aux | grep "train.py --profile ablation_20260407_amdqfd" | grep -v grep | wc -l)

    # 检查模型文件是否存在
    NOAM_PT=$(ls "$PROJ"/runs/abl_amdqfd_dmdqn_noAM/train_*/models/realmap_a/cnn-ddqn.pt 2>/dev/null | head -1)
    NODQFD_PT=$(ls "$PROJ"/runs/abl_amdqfd_dmdqn_noDQfD/train_*/models/realmap_a/cnn-ddqn.pt 2>/dev/null | head -1)

    # 条件：进程全部退出 + 两个模型文件都存在
    if [ "$N_TRAIN" -eq 0 ] && [ -n "$NOAM_PT" ] && [ -n "$NODQFD_PT" ]; then
        echo "[$(ts)] Both training complete!"
        echo "  noAM  model: $NOAM_PT ($(stat -c%s "$NOAM_PT" 2>/dev/null || stat -f%z "$NOAM_PT") bytes)"
        echo "  noDQfD model: $NODQFD_PT ($(stat -c%s "$NODQFD_PT" 2>/dev/null || stat -f%z "$NODQFD_PT") bytes)"
        break
    fi

    # 每 5 分钟报告一次进度
    echo -n "[$(ts)] training procs=$N_TRAIN | "
    [ -n "$NOAM_PT" ]  && echo -n "noAM=DONE " || echo -n "noAM=running "
    [ -n "$NODQFD_PT" ] && echo -n "noDQfD=DONE" || echo -n "noDQfD=running"
    echo ""

    sleep 300
done

# ---- Phase 2: 验证 Full 模型也在 ----
FULL_PT=$(ls "$PROJ"/runs/abl_8var_cnn_ddqn_munch_duel/train_*/models/realmap_a/cnn-ddqn.pt 2>/dev/null | head -1)
if [ -z "$FULL_PT" ]; then
    echo "[$(ts)] ERROR: Full model not found! Aborting."
    exit 1
fi
echo "[$(ts)] Full model: $FULL_PT"

# ---- Phase 3: 启动 6 轮推理（顺序执行，稳定可靠） ----
PROFILES=(
    ablation_20260407_amdqfd_dmdqn_infer_full_sr_short
    ablation_20260407_amdqfd_dmdqn_infer_full_sr_long
    ablation_20260407_amdqfd_dmdqn_infer_noAM_sr_short
    ablation_20260407_amdqfd_dmdqn_infer_noAM_sr_long
    ablation_20260407_amdqfd_dmdqn_infer_noDQfD_sr_short
    ablation_20260407_amdqfd_dmdqn_infer_noDQfD_sr_long
)

TOTAL=${#PROFILES[@]}
OK=0
FAIL=0

for i in "${!PROFILES[@]}"; do
    p="${PROFILES[$i]}"
    n=$((i+1))
    echo ""
    echo "[$(ts)] [$n/$TOTAL] Starting inference: $p"
    $CONDA run --cwd "$PROJ" -n $ENV python infer.py --profile "$p"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(ts)] [$n/$TOTAL] SUCCESS: $p"
        ((OK++))
    else
        echo "[$(ts)] [$n/$TOTAL] FAILED (exit=$rc): $p"
        ((FAIL++))
    fi
done

# ---- Phase 4: 汇总 ----
echo ""
echo "[$(ts)] === All inference done: $OK success, $FAIL failed ==="

# 列出产出的 CSV 文件
echo "[$(ts)] Output CSVs:"
for p in "${PROFILES[@]}"; do
    OUT_NAME=$(python3 -c "import json; d=json.load(open('$PROJ/configs/${p}.json')); print(d['infer']['out'])")
    CSV=$(ls "$PROJ"/runs/"$OUT_NAME"/*/table2_kpis.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        ROWS=$(wc -l < "$CSV")
        echo "  $OUT_NAME: $CSV ($ROWS rows)"
    else
        echo "  $OUT_NAME: NOT FOUND"
    fi
done

echo "[$(ts)] === Script finished ==="
