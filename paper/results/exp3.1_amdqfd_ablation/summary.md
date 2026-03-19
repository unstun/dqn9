# 实验 3.1：AM × DQfD 组件消融（3 变体，带轨迹数据）

> 与 exp3 结果一致（SR 相同，Quality 计算时间有微小波动），额外保存了逐步轨迹 traces。

## 实验设计

- **目标**：验证动作掩码（AM）和 DQfD 预训练各自的贡献
- **基底**：CNN-DDQN+MD (reward_k_t=0.2, EDT diag)
- **训练**：10000 episodes, seed=0
- **推理**：Seeds 100/200/300/400/500/600/700，每 seed 7 runs（700=8），共 50 runs/variant/distance
- **设计原则**：训练时消融，推理统一带 mask（隔离训练时贡献）
- **配置**：`configs/ablation_20260315_amdqfd_*.json`
- **重跑日期**：2026-03-16（加 `--save-traces` 重跑）

## 变体设计

| 变体 | AM (shield+TD mask) | DQfD (prefill+pretrain+expert_exploration) | 训练来源 |
|------|--------------------|--------------------------------------------|----------|
| Full (AM+DQfD) | ON | ON | 复用 `abl_diag10k_kt02_cnn_ddqn_md` |
| w/o DQfD | ON | OFF | `abl_amdqfd_noDQfD` |
| w/o AM | OFF | ON | `abl_amdqfd_noAM` |

## 成功率（SR，50 runs）

| 变体 | Long SR | Short SR |
|------|---------|----------|
| **Full (AM+DQfD)** | **92%** (46/50) | **90%** (45/50) |
| w/o AM | 72% (36/50) | 86% (43/50) |
| w/o DQfD | 56% (28/50) | 80% (40/50) |

## 路径质量（Quality，三变体均成功的 runs）

### Long distance — 19/50 runs 三变体全成功

| 指标 | Full | w/o AM | w/o DQfD |
|------|------|--------|----------|
| PathLen | **24.652m** | 25.317m | 25.831m |
| Curvature | **0.1412** | 0.1537 | 0.1812 |
| Time | **0.497s** | 0.507s | 0.691s |

### Short distance — 32/50 runs 三变体全成功

| 指标 | Full | w/o AM | w/o DQfD |
|------|------|--------|----------|
| PathLen | 8.809m | **8.778m** | 9.048m |
| Curvature | 0.1685 | **0.1500** | 0.1717 |
| Time | 0.261s | **0.257s** | 0.341s |

## 与 exp3 对比

| 指标 | exp3 (旧) | exp3.1 (.1) | 差异 |
|------|-----------|-------------|------|
| Long SR (Full/noAM/noDQfD) | 92/72/56% | 92/72/56% | 相同 |
| Short SR | 90/86/80% | 90/86/80% | 相同 |
| Long Quality PL (Full) | 24.652m | 24.652m | 相同 |
| Long Quality Time (Full) | 0.490s | 0.497s | +0.007s |
| Short Quality Time (Full) | 0.287s | 0.261s | -0.026s |

> 计算时间微小差异来自服务器负载波动。路径长度/曲率/排名不变。

## 结论

- **DQfD 预训练贡献最大**：移除后 Long SR 下降 36pp（92% → 56%），Quality 路径长度 +1.18m
- **AM 显著辅助**：移除后 Long SR 下降 20pp（92% → 72%），Quality 路径长度 +0.67m
- **两者协同最优**：Full 在 Long 距离 SR 和 Quality 上全面领先
- **Short 距离**：w/o AM 在 Quality 上略优（路径长度 -0.03m），但 SR 低 4pp

## Trace 数据说明

每个推理 run 在 `traces/` 子目录下保存逐步轨迹 CSV，列包括：

| 列名 | 说明 |
|------|------|
| step | 时间步 |
| x_m, y_m | 车辆位置（米） |
| theta_rad | 航向角（弧度） |
| v_m_s | 线速度 |
| delta_rad | 前轮转角 |
| action | 动作索引 |
| od_m | 最近障碍物距离（EDT 双圆检测） |
| collision | 是否碰撞 |
| reached | 是否到达目标 |
| reward | 即时奖励 |

## 详细日志

见 `runs/ablation_logs/ablation_20260316_amdqfd.md`

## 数据路径

- `abl_amdqfd_infer_full.1/` → Full (AM+DQfD) 推理结果（含 traces）
- `abl_amdqfd_infer_noAM.1/` → w/o AM 推理结果（含 traces）
- `abl_amdqfd_infer_noDQfD.1/` → w/o DQfD 推理结果（含 traces）
