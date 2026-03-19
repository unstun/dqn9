# 实验 3：AM × DQfD 组件消融（3 变体）

## 实验设计

- **目标**：验证动作掩码（AM）和 DQfD 预训练各自的贡献
- **基底**：CNN-DDQN+MD (reward_k_t=0.2, EDT diag)
- **训练**：10000 episodes, seed=0
- **推理**：Seeds 100/200/300/400/500/600/700，50 runs/variant/distance
- **设计原则**：训练时消融，推理统一带 mask（隔离训练时贡献）
- **配置**：`configs/ablation_20260315_amdqfd_*.json`

## 变体设计

| 变体 | AM (shield+TD mask) | DQfD (prefill+pretrain+expert_exploration) | 训练来源 |
|------|--------------------|--------------------------------------------|----------|
| Full (AM+DQfD) | ON | ON | 复用 `abl_diag10k_kt02_cnn_ddqn_md` |
| w/o DQfD | ON | OFF | `abl_amdqfd_noDQfD` |
| w/o AM | OFF | ON | `abl_amdqfd_noAM` |

## 结果

### 成功率（SR，50 runs）

| 变体 | Long SR | Short SR |
|------|---------|----------|
| **Full (AM+DQfD)** | **92%** | **90%** |
| w/o AM | 72% | 86% |
| w/o DQfD | 56% | 80% |

### 路径质量（Quality，三变体均成功的 runs）

| 距离 | 指标 | Full | w/o AM | w/o DQfD |
|------|------|------|--------|----------|
| Long (19 runs) | PathLen | **24.652m** | 25.317m | 25.831m |
| Long | Curvature | **0.1412** | 0.1537 | 0.1812 |
| Long | Time | **0.490s** | 0.523s | 0.648s |
| Short (32 runs) | PathLen | 8.809m | **8.778m** | 9.048m |
| Short | Curvature | 0.1685 | **0.1500** | 0.1718 |
| Short | Time | 0.287s | **0.258s** | 0.370s |

## 结论

- **DQfD 预训练贡献最大**：移除后 Long SR 下降 36pp（92% → 56%）
- **AM 显著辅助**：移除后 Long SR 下降 20pp（92% → 72%）
- **两者协同最优**：Full 在 Long 距离 SR 和 Quality 上全面领先

## 详细日志

见 `runs/ablation_logs/ablation_20260316_amdqfd.md`

## 数据路径

- `abl_amdqfd_infer_full/` — Full (AM+DQfD) 推理结果
- `abl_amdqfd_infer_noAM/` — w/o AM 推理结果
- `abl_amdqfd_infer_noDQfD/` — w/o DQfD 推理结果
