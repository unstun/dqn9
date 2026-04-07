#!/bin/bash
# ============================================================
# §3.5 全因子组件消融：训练完成后自动推理
# 等待 6 个新训练（DQN/Duel/M-DQN × noAM/noDQfD）结束
# → 启动 18 轮推理（DQN/Duel/M-DQN × Full/noAM/noDQfD × Short/Long）
# 用法: nohup bash scripts/auto_infer_35_full.sh > runs/auto_infer_35_full.log 2>&1 &
# ============================================================

PROJ=~/DQN9
CONDA=~/miniconda3/bin/conda
ENV=ros2py310

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] === §3.5 full-factor auto-inference script started ==="

# ---- Phase 1: 等待 6 个新训练完成 ----
echo "[$(ts)] Waiting for 6 new trainings to finish..."

# 6 个待训变体
NEW_VARIANTS=(
    abl_amdqfd_dqn_noAM
    abl_amdqfd_dqn_noDQfD
    abl_amdqfd_dueldqn_noAM
    abl_amdqfd_dueldqn_noDQfD
    abl_amdqfd_mdqn_noAM
    abl_amdqfd_mdqn_noDQfD
)

while true; do
    # 检查训练进程
    N_TRAIN=$(ps aux | grep "train.py --profile ablation_20260408_amdqfd" | grep -v grep | wc -l)

    # 检查 6 个模型文件
    N_DONE=0
    for v in "${NEW_VARIANTS[@]}"; do
        PT=$(ls "$PROJ"/runs/"$v"/train_*/models/realmap_a/cnn-ddqn.pt 2>/dev/null | head -1)
        [ -n "$PT" ] && N_DONE=$((N_DONE+1))
    done

    if [ "$N_TRAIN" -eq 0 ] && [ "$N_DONE" -eq 6 ]; then
        echo "[$(ts)] All 6 trainings complete!"
        for v in "${NEW_VARIANTS[@]}"; do
            PT=$(ls "$PROJ"/runs/"$v"/train_*/models/realmap_a/cnn-ddqn.pt 2>/dev/null | head -1)
            SZ=$(stat -c%s "$PT" 2>/dev/null || stat -f%z "$PT")
            echo "  $v: $PT ($SZ bytes)"
        done
        break
    fi

    echo "[$(ts)] training procs=$N_TRAIN | models ready=$N_DONE/6"
    sleep 300
done

# ---- Phase 2: 验证 4 个 Full 模型（8变体复用）也在 ----
FULL_DIRS=(
    abl_8var_cnn_ddqn
    abl_8var_cnn_ddqn_duel
    abl_8var_cnn_ddqn_munch
    abl_8var_cnn_ddqn_munch_duel
)
for d in "${FULL_DIRS[@]}"; do
    PT=$(ls "$PROJ"/runs/"$d"/train_*/models/realmap_a/cnn-ddqn.pt 2>/dev/null | head -1)
    if [ -z "$PT" ]; then
        echo "[$(ts)] ERROR: 8var Full model not found: $d. Aborting."
        exit 1
    fi
    echo "[$(ts)] Full reuse: $d → $PT"
done

# ---- Phase 3: 启动 18 轮推理（顺序执行） ----
PROFILES=(
    ablation_20260408_amdqfd_dqn_infer_full_sr_short
    ablation_20260408_amdqfd_dqn_infer_full_sr_long
    ablation_20260408_amdqfd_dqn_infer_noAM_sr_short
    ablation_20260408_amdqfd_dqn_infer_noAM_sr_long
    ablation_20260408_amdqfd_dqn_infer_noDQfD_sr_short
    ablation_20260408_amdqfd_dqn_infer_noDQfD_sr_long
    ablation_20260408_amdqfd_dueldqn_infer_full_sr_short
    ablation_20260408_amdqfd_dueldqn_infer_full_sr_long
    ablation_20260408_amdqfd_dueldqn_infer_noAM_sr_short
    ablation_20260408_amdqfd_dueldqn_infer_noAM_sr_long
    ablation_20260408_amdqfd_dueldqn_infer_noDQfD_sr_short
    ablation_20260408_amdqfd_dueldqn_infer_noDQfD_sr_long
    ablation_20260408_amdqfd_mdqn_infer_full_sr_short
    ablation_20260408_amdqfd_mdqn_infer_full_sr_long
    ablation_20260408_amdqfd_mdqn_infer_noAM_sr_short
    ablation_20260408_amdqfd_mdqn_infer_noAM_sr_long
    ablation_20260408_amdqfd_mdqn_infer_noDQfD_sr_short
    ablation_20260408_amdqfd_mdqn_infer_noDQfD_sr_long
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
        OK=$((OK+1))
    else
        echo "[$(ts)] [$n/$TOTAL] FAILED (exit=$rc): $p"
        FAIL=$((FAIL+1))
    fi
done

# ---- Phase 4: 汇总 ----
echo ""
echo "[$(ts)] === All inference done: $OK success, $FAIL failed ==="

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
