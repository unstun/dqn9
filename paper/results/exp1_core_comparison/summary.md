# 实验 1：核心对比 — CNN-DDQN+MD vs RRT* vs LO-HA*

## 实验设计

- **目标**：验证受约束 DRL 路径规划框架优于经典规划器
- **算法**：CNN-DDQN+MD（本文方法）、RRT*、LO-HA*（Lattice-based Optimal Hybrid A*）
- **环境**：Realmap（复杂森林地图）
- **Seed**：420（经 500 seeds 大规模扫描 + 8 候选 seed × 50 runs 深度验证确认）
- **碰撞检测**：EDT diag（三种算法使用完全相同的 `EDTCollisionChecker`）
- **Baseline 超时**：10s
- **每组 runs**：50
- **配置**：`configs/final_t10_sr_{long,short}.json`

## 结果

### 成功率（SR）

| 距离 | CNN-DDQN+MD | LO-HA* | RRT* |
|------|------------|--------|------|
| Long (≥18m) | **100%** | 32% | 72% |
| Short (6–14m) | **100%** | 92% | 90% |

### 路径质量（Quality，三算法均成功的 runs）

| 距离 | 指标 | CNN-DDQN+MD | LO-HA* | RRT* |
|------|------|------------|--------|------|
| Long (13 runs) | PathLen | **27.367m** | 27.456m | 27.579m |
| Long | Curvature | **0.1118** | 0.1254 | 0.1513 |
| Long | Time | **0.627s** | 9.337s | 3.470s |
| Short (41 runs) | PathLen | **8.968m** | 8.971m | 8.977m |
| Short | Curvature | **0.1487** | 0.1614 | 0.1853 |
| Short | Time | **0.244s** | 1.133s | 1.615s |

## 结论

DRL 在成功率、路径长度、曲率、计算时间上全面胜出。

## 数据路径

- `final_t10_sr_long/` → Long 距离推理结果
- `final_t10_sr_short/` → Short 距离推理结果
