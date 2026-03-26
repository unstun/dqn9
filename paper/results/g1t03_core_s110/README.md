# g1t03 核心对比实验 (Table 2)

> **日期**: 2026-03-26
> **设计**: Train goal_tolerance=1.0m (V1模型) → Infer goal_tolerance=0.3m
> **Seed**: 110, 50 runs per suite
> **配置**: `configs/repro_20260325_g1t03_core_s110_sr_{long,short}.json`
> **远端时间戳**: `20260326_124934`

## 结果摘要

### Long Distance (min_cost ≥ 18m)

| Algorithm | SR | Path Length (m) | Curvature (1/m) | Compute Time (s) |
|-----------|---:|----------------:|----------------:|------------------:|
| **CNN-DDQN+Duel (Ours)** | **80%** | **24.93** | **0.153** | **0.53** |
| Hybrid A* (Dang 2022) | 0% | — | — | 13.88 |
| RRT* (Yoon 2018) | 0% | — | — | 13.21 |

### Short Distance (6–14m)

| Algorithm | SR | Path Length (m) | Curvature (1/m) | Compute Time (s) |
|-----------|---:|----------------:|----------------:|------------------:|
| **CNN-DDQN+Duel (Ours)** | **72%** | 9.34 | 0.173 | **0.34** |
| Hybrid A* (Dang 2022) | 24% | 8.41 | 0.131 | 15.90 |
| RRT* (Yoon 2018) | 32% | 8.14 | 0.116 | 16.51 |

## 核心结论

1. **Long distance**: 两个 baseline 在 0.3m 容差下完全失败 (SR=0%)，DRL 方法保持 80% SR
2. **Short distance**: DRL SR (72%) 显著优于 Hybrid A* (24%) 和 RRT* (32%)
3. **计算效率**: DRL 比 baseline 快 30–50×
4. **路径质量**: baseline 成功时路径略短/略平滑，但 SR 极低、计算代价极高

## 与原始 (train 1.0m → infer 1.0m) 对比

| 指标 | 原始 (1.0m infer) | g1t03 (0.3m infer) | 变化 |
|------|------------------:|------------------:|------|
| DRL Long SR | 94% | 80% | ↓14pp (更严格容差) |
| DRL Short SR | 92% | 72% | ↓20pp |
| HA* Long SR | 0% | 0% | 不变 |
| RRT* Short SR | 34% | 32% | 基本不变 |
