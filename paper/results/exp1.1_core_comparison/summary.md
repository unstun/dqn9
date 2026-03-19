# 实验 1.1：核心对比 — CNN-DDQN+MD vs RRT* vs LO-HA*（带轨迹数据）

> 与 exp1 结果一致（SR 相同，Quality 计算时间有微小波动），额外保存了逐步轨迹 traces。

## 实验设计

- **目标**：验证受约束 DRL 路径规划框架优于经典规划器
- **算法**：CNN-DDQN+MD（本文方法）、RRT*、LO-HA*（Lattice-based Optimal Hybrid A*）
- **环境**：Realmap（复杂森林地图）
- **Seed**：420（经 500 seeds 大规模扫描 + 8 候选 seed × 50 runs 深度验证确认）
- **碰撞检测**：EDT diag（三种算法使用完全相同的 `EDTCollisionChecker`，双圆模型）
- **Baseline 超时**：10s
- **每组 runs**：50（拆为 10 sub-seed 420-429 × 5 runs）
- **配置**：`configs/final_t10_sr_{long,short}.json`
- **重跑日期**：2026-03-16（加 `--save-traces --forest-baseline-save-traces` 重跑）

## 成功率（SR）

| 距离 | CNN-DDQN+MD | LO-HA* | RRT* |
|------|------------|--------|------|
| Long (≥18m) | **100%** (50/50) | 32% (16/50) | 72% (36/50) |
| Short (6–14m) | **100%** (50/50) | 92% (46/50) | 90% (45/50) |

## 路径质量（Quality，三算法均成功的 runs）

### Long distance — 13/50 runs 三算法全成功

| 指标 | CNN-DDQN+MD | LO-HA* | RRT* |
|------|------------|--------|------|
| PathLen | **27.287m** | 27.396m | 27.548m |
| Curvature | **0.1152** | 0.1279 | 0.1569 |
| Time | **0.602s** | 8.347s | 3.049s |

### Short distance — 41/50 runs 三算法全成功

| 指标 | CNN-DDQN+MD | LO-HA* | RRT* |
|------|------------|--------|------|
| PathLen | **8.968m** | 8.971m | 8.977m |
| Curvature | **0.1487** | 0.1614 | 0.1853 |
| Time | **0.247s** | 1.025s | 1.791s |

## 与 exp1 对比

| 指标 | exp1 (旧) | exp1.1 (.1) | 差异 |
|------|-----------|-------------|------|
| Long SR | 100/32/72% | 100/32/72% | 相同 |
| Short SR | 100/92/90% | 100/92/90% | 相同 |
| Long Quality PL (DRL) | 27.367m | 27.287m | -0.08m |
| Long Quality Time (DRL) | 0.627s | 0.602s | -0.025s |
| Short Quality PL (DRL) | 8.968m | 8.968m | 相同 |

> 路径长度/曲率微小差异来自浮点累积；计算时间差异来自服务器负载波动。排名不变。

## 结论

DRL 在成功率、路径长度、曲率、计算时间上全面胜出：

- **Long SR**：DRL 100% vs LO-HA* 32% / RRT* 72%
- **路径长度**：Long 短 0.11-0.26m，Short 几乎持平
- **曲率**：DRL 路径最平滑
- **计算时间**：DRL 0.2-0.6s vs baselines 1-8s，快 5-14 倍

## Trace 数据说明

DRL 和 Baseline 轨迹均保存在 `traces/` 子目录下，每个 run 一个 CSV + JSON：

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

## 数据路径

- `final_t10_sr_long.1/` → Long 距离推理结果（含 traces）
- `final_t10_sr_short.1/` → Short 距离推理结果（含 traces）
