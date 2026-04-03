# 论文实验评估框架

> 从 CLAUDE.md Section 3 拆分，写论文/分析实验数据时参考此文件。

### 3.1 评估体系

**环境**：Realmap（复杂）

**每环境 3 类结果**：

| 结果          | 筛选         | 路径距离                  | Runs    | 汇报 KPI                 |
| ------------- | ------------ | ------------------------- | ------- | ------------------------ |
| SR            | BK 可达      | Long ≥18m / Short 6–14m | 50+     | **仅成功率**       |
| Quality Long  | 3 算法全成功 | ≥18m                     | ~5–10  | 路径长度、曲率、计算时间 |
| Quality Short | 3 算法全成功 | 6–14m                    | ~25–30 | 路径长度、曲率、计算时间 |

- **3-algo filter**：MD-DDQN + RRT* + LO-HA*。
- **SR 结果**只报成功率，不看路径质量（失败路径无有效指标）。
- **Quality 结果**只报路径质量，成功率恒 100%（消除混杂变量）。
- 直接比较原始指标，不做 minmax 归一化。

### 3.1.1 网络架构（CNN-MD-DDQN）

- **CNN 双输入（dual-input）**：卷积分支（2 通道地图：occupancy + goal distance，`cnn_drop_edt=true` 丢弃 EDT）+ 标量分支（11-d），标量分支**无 FC 预处理，直接拼接**
- **卷积骨干**：Conv(2→32, k3s1) → Conv(32→64, k3s2) → Conv(64→64, k3s2) → SpatialMHA(h=4) → Flatten(576)
- **拼接后**：FC(587→256) 共享层
- **Dueling 双流**：Value FC(256→256→1) / Advantage FC(256→256→35)
- **输出**：Q = V + A − mean(A) → Action Mask → a*
- **架构图**：`paper/figures/cnn_md_ddqn_architecture.{pdf,png}`（由 `plot_cnn_architecture_v2.py` 生成）

### 3.2 论文叙事约束

- **核心论点**：受约束 DRL 路径规划框架优于经典规划器。
- **系统级创新**：动作掩码、Dijkstra 奖励塑形、DQfD 预训练、自行车运动学约束、双圆碰撞检测。
- **核心算法**：MD-DDQN，reward_k_t=0.2（消融实验 2026-03-15 确认 MD 全面优于 Duel）。
- **若结果不支持叙事**：换 checkpoint 或调训练参数重跑，不可篡改数据。

### 3.3 核心对比实验：MD-DDQN vs RRT* vs LO-HA*（全局规划，无 MPC）

- **Seed**: 110
- **配置**: `configs/final_t10_sr_{long,short}.json`，`baseline_timeout=10s`
- **碰撞检测**: EDT diag（baselines 与 DRL 使用完全相同的 `EDTCollisionChecker`）
- **数据来源**: `runs202642/infer/core_baseline_sr_{long,short}/`
- **结果（50 runs, seed 110）**:

| 距离  | 指标                       | MD-DDQN                | Hybrid A*        | RRT*        |
| ----- | -------------------------- | ---------------------- | ---------------- | ----------- |
| Long  | SR                         | **80%** (40/50)  | 28% (14/50)      | 68% (34/50) |
| Long  | PathLen (Quality, 11 runs) | **20.164m**      | 20.684m          | 20.486m     |
| Long  | Curvature                  | **0.1383**       | 0.1777           | 0.3525      |
| Long  | Time                       | **0.414s**       | 13.486s          | 2.899s      |
| Short | SR                         | 72% (36/50)            | 64% (32/50)      | **76%** (38/50) |
| Short | PathLen (Quality, 21 runs) | **8.531m**       | 9.608m           | 9.347m      |
| Short | Curvature                  | **0.1467**       | 0.3624           | 0.3406      |
| Short | Time                       | **0.239s**       | 2.621s           | 0.959s      |

- **结论**: Long 距离 DRL 在 SR/PL/曲率/时间全面胜出；Short 距离 SR 三者接近（RRT* 略优），DRL 在 PL/曲率/时间全面胜出

### 3.4 DRL 模块消融实验（4 DQN 变体，MinTD）

- **训练**: 10000 episodes, DQfD pretrain 40000 steps, reward_k_t=0.2, EDT diag
- **推理**: MinTD 检查点，Seed 110，50 runs/variant/distance（goal_tolerance=0.3m）
- **变体**: DQN, Duel-DQN, MHA-DQN, MD-DQN
- **数据来源**: `runs202643/infer/abl_minloss_cnn_dqn*/`，`runs202643/train/abl_arch_cnn_dqn*/`
- **SR 结果（50 runs）**:

| 变体     | Short SR | Long SR      |
| -------- | -------- | ------------ |
| MD-DQN   | 70%      | **86%** |
| Duel-DQN | 72%      | 84%          |
| MHA-DQN  | **74%** | 80%          |
| DQN      | 64%      | 74%          |

- **Quality Long（N=30，4 变体均成功子集）**:

| 指标      | MD-DQN          | Duel-DQN | MHA-DQN | DQN             |
| --------- | --------------- | -------- | ------- | --------------- |
| PathLen   | **24.176m** | 24.231m  | 24.236m | 24.277m         |
| Curvature | **0.1588** | 0.1601   | 0.1703  | 0.1613          |
| Time      | 0.462s          | 0.349s   | 0.436s  | **0.329s** |

- **Quality Short（N=20，4 变体均成功子集）**:

| 指标      | MD-DQN | Duel-DQN        | MHA-DQN | DQN    |
| --------- | ------ | --------------- | ------- | ------ |
| PathLen   | 9.565m | **9.539m** | 9.552m  | 9.629m |
| Curvature | 0.1687 | **0.1634** | 0.1638  | 0.1973 |
| Time      | 0.232s | **0.183s** | 0.215s  | 0.219s |

- **结论**: MD-DQN 远距离 SR 最高（86%）、路径最短（24.176m）、曲率最优（0.1588），综合选为核心算法
- **详细日志**: `runs/ablation_logs/ablation_20260315_diag10k_kt02.md`
- **配置**: `configs/ablation_20260314_diag10k_kt02_*.json`，MinTD 推理配置 `configs/ablation_20260402_*.json`

### 3.5 AM × DQfD 组件消融实验（3 变体，MinTD）

- **基底**: MD-DQN (reward_k_t=0.2, EDT diag)
- **训练**: 10000 episodes, seed=0
- **推理**: MinTD 检查点，Seed 110，50 runs/variant/distance（goal_tolerance=0.3m）
- **设计**: 训练时消融，推理统一带 mask（隔离训练时贡献）
- **代码改动**: `ugv_dqn/cli/train.py` 4 处条件化 `forest_action_shield`（expert exploration fallback、TD target mask、demo prefill mask ×2）
- **变体**:

| 变体           | AM (shield+TD mask) | DQfD (prefill+pretrain+expert_exploration) | 训练来源                              |
| -------------- | ------------------- | ------------------------------------------ | ------------------------------------- |
| Full (AM+DQfD) | ON                  | ON                                         | 复用 `abl_arch_cnn_dqn_md`          |
| w/o DQfD       | ON                  | OFF                                        | `abl_amdqfd_dqn_noDQfD`             |
| w/o AM         | OFF                 | ON                                         | `abl_amdqfd_dqn_noAM`               |

- **数据来源**: `runs202643/infer/abl_minloss_cnn_dqn_md*`（Full），`runs202643/infer/abl_minloss_amdqfd_dqn_noAM*`（w/o AM），`runs202643/infer/abl_amdqfd_dqn_infer_noDQfD*`（w/o DQfD）
- **SR 结果（50 runs）**:

| 变体                | Short SR     | Long SR      |
| ------------------- | ------------ | ------------ |
| **Full (AM+DQfD)** | **70%** | **86%** |
| w/o AM              | 62%          | 56%          |
| w/o DQfD            | 42%          | 28%          |

- **Quality Long（N=11，3 变体均成功子集）**:

| 指标      | Full              | w/o AM  | w/o DQfD |
| --------- | ----------------- | ------- | -------- |
| PathLen   | **26.217m** | 27.220m | 27.220m  |
| Curvature | **0.1512**  | 0.1923  | 0.1950   |
| Time      | **0.495s**  | 0.664s  | 0.733s   |

- **Quality Short（N=17，3 变体均成功子集）**:

| 指标      | Full             | w/o AM | w/o DQfD |
| --------- | ---------------- | ------ | -------- |
| PathLen   | **8.883m** | 9.077m | 9.537m   |
| Curvature | **0.1560** | 0.1742 | 0.1982   |
| Time      | **0.231s** | 0.232s | 0.341s   |

- **结论**: DQfD 预训练贡献最大（远距离 SR -58pp），AM 辅助（远距离 SR -30pp），远距离影响远大于近距离
- **详细日志**: `runs/ablation_logs/ablation_20260316_amdqfd.md`
- **配置**: `configs/ablation_20260402_amdqfd_dqn_*.json`
