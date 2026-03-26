# g1t03 跨容差消融实验结果

> Train goal_tolerance = 1.0m (V1 模型) → Infer goal_tolerance = 0.3m
> 日期: 2026-03-25/26

## 实验设计

使用 1.0m 容差训练的模型，在 0.3m 容差下推理，验证消融结论的稳健性。

## V1 模型映射

| 消融类型 | 实验目录 | V1 train 时间戳 |
|---------|---------|----------------|
| 架构 8 CNN + MLP | `abl_diag10k_kt02_cnn_{ddqn,dqn}_{,duel,mha,md}` | `train_20260315_120749` |
| Scalar-Only | `abl_scalar_only` | `train_20260323_194448` |
| 分辨率 n8/n16/n24 | `abl_resolution_{n8,n16,n24}` | `train_20260319_111513` |
| AM×DQfD full | `abl_diag10k_kt02_cnn_ddqn_md` | `train_20260315_120749` |
| AM×DQfD noDQfD/noAM | `abl_amdqfd_{noDQfD,noAM}` | `train_20260316_001347` |

## 目录结构

```
g1t03_crosstol/
├── summary.md                     # 完整分析报告
├── README.md                      # 本文件
├── architecture/                  # 架构消融 (11 变体, seed=42, 50 runs)
│   ├── sr_table.csv               # 成功率汇总
│   ├── quality_long_table.csv     # Long 路径质量
│   ├── quality_short_table.csv    # Short 路径质量
│   └── raw/                       # 原始 per-run CSV (20 文件)
├── resolution/                    # 分辨率消融 (3 变体, seed=200, 50 runs)
│   ├── sr_table.csv
│   ├── quality_long_table.csv
│   ├── quality_short_table.csv
│   └── raw/                       # 原始 CSV (6 文件)
└── amdqfd/                        # AM×DQfD 消融 (3 变体, seeds=100-700, 50 runs)
    ├── sr_table.csv
    ├── quality_long_table.csv
    ├── quality_short_table.csv
    └── raw/                       # 原始 CSV (42 文件, 7 seeds × 3 变体 × 2 距离)
```

## 核心结论

所有消融排序与 Train 1.0m → Infer 1.0m **完全一致**:
- **架构**: MD-DDQN Long SR (84%) + Quality (PL 26.016m, κ 0.1413) 双冠
- **分辨率**: N=8 崩溃 (18%), N≥16 稳健 (78%)
- **AM×DQfD**: Full > w/o AM > w/o DQfD, DQfD 贡献最大

## 配置与脚本

- 推理配置: `configs/ablation_20260325_g1t03_*.json` (32 个)
- 批量脚本: `scripts/run_g1t03_infer.sh`
- 远端输出: `runs/*_g1t03/`
