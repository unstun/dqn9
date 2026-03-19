# 实验 2.1：DRL 模块消融（10 变体，带轨迹数据）

> 与 exp2 结果完全一致，额外保存了逐步轨迹 traces（x, y, theta, od_m, collision 等）。

## 实验设计

- **目标**：确认最优网络架构组合
- **基底**：DQfD pretrain 40000 steps, reward_k_t=0.2, EDT diag 碰撞检测
- **训练**：10000 episodes, seed=0
- **推理**：Seeds 42/142/242/342/442/542/642，每 seed 7 runs（642=8），共 50 runs/variant/distance
- **配置**：`configs/ablation_20260314_diag10k_kt02_*.json`
- **服务器**：uhost RTX 4090, 16×AMD EPYC 7543
- **重跑日期**：2026-03-16（加 `--save-traces` 重跑，结果可复现）

## 变体列表（10 个）

| # | 变体 | 编码器 | 目标网络 | 输出头 | 说明 |
|---|------|--------|----------|--------|------|
| 1 | CNN-DQN | CNN | DQN | Linear | 基线 CNN |
| 2 | CNN-DQN+Duel | CNN | DQN | Dueling | Dueling 架构 |
| 3 | CNN-DQN+MHA | CNN | DQN | MHA | 多头注意力 |
| 4 | CNN-DQN+MD | CNN | DQN | Multi-Discrete | 多离散动作分解 |
| 5 | CNN-DDQN | CNN | Double DQN | Linear | 双网络去偏 |
| 6 | CNN-DDQN+Duel | CNN | Double DQN | Dueling | |
| 7 | CNN-DDQN+MHA | CNN | Double DQN | MHA | |
| 8 | CNN-DDQN+MD | CNN | Double DQN | Multi-Discrete | **论文核心算法** |
| 9 | MLP-DQN | MLP | DQN | Linear | 无空间编码 |
| 10 | MLP-DDQN | MLP | Double DQN | Linear | |

## SR 结果（成功率，50 runs）

### Long distance (≥18m)

| 排名 | 变体 | 成功 | SR |
|------|------|-----|----|
| 1 | CNN-DQN | 48/50 | **96.0%** |
| 2 | CNN-DDQN+MD | 47/50 | 94.0% |
| 2 | CNN-DQN+Duel | 47/50 | 94.0% |
| 4 | CNN-DQN+MHA | 46/50 | 92.0% |
| 5 | CNN-DDQN | 45/50 | 90.0% |
| 5 | CNN-DQN+MD | 45/50 | 90.0% |
| 7 | CNN-DDQN+Duel | 44/50 | 88.0% |
| 7 | MLP-DQN | 44/50 | 88.0% |
| 7 | MLP-DDQN | 44/50 | 88.0% |
| 10 | CNN-DDQN+MHA | 41/50 | 82.0% |

### Short distance (6–14m)

| 排名 | 变体 | 成功 | SR |
|------|------|-----|----|
| 1 | CNN-DQN+Duel | 47/50 | **94.0%** |
| 2 | CNN-DDQN+Duel | 46/50 | 92.0% |
| 2 | CNN-DDQN+MHA | 46/50 | 92.0% |
| 2 | CNN-DDQN+MD | 46/50 | 92.0% |
| 5 | CNN-DDQN | 45/50 | 90.0% |
| 5 | CNN-DQN | 45/50 | 90.0% |
| 5 | CNN-DQN+MD | 45/50 | 90.0% |
| 8 | CNN-DQN+MHA | 44/50 | 88.0% |
| 9 | MLP-DDQN | 43/50 | 86.0% |
| 10 | MLP-DQN | 42/50 | 84.0% |

## Quality 结果（全 10 变体均成功的 runs）

筛选条件：同一 (seed, run_index) 下所有 10 个变体均成功。

### Long distance — 33/50 runs 全成功

| 排名 | 变体 | N | 平均路径长度(m) | 平均曲率(1/m) | 平均计算时间(s) |
|------|------|---|----------------|--------------|----------------|
| 1 | **CNN-DDQN+MD** | 33 | **26.365** | 0.1306 | 0.507 |
| 2 | CNN-DDQN+Duel | 33 | 26.373 | 0.1377 | 0.430 |
| 3 | MLP-DDQN | 33 | 26.422 | 0.1337 | **0.288** |
| 4 | CNN-DQN+Duel | 33 | 26.428 | 0.1345 | 0.354 |
| 5 | CNN-DQN | 33 | 26.436 | 0.1426 | 0.346 |
| 6 | CNN-DQN+MHA | 33 | 26.440 | 0.1344 | 0.463 |
| 7 | CNN-DQN+MD | 33 | 26.454 | 0.1342 | 0.537 |
| 8 | MLP-DQN | 33 | 26.459 | 0.1452 | 0.279 |
| 9 | CNN-DDQN | 33 | 26.492 | 0.1369 | 0.323 |
| 10 | CNN-DDQN+MHA | 33 | 26.598 | 0.1508 | 0.460 |

### Short distance — 29/50 runs 全成功

| 排名 | 变体 | N | 平均路径长度(m) | 平均曲率(1/m) | 平均计算时间(s) |
|------|------|---|----------------|--------------|----------------|
| 1 | **CNN-DDQN+MD** | 29 | **8.604** | 0.1711 | 0.224 |
| 2 | CNN-DDQN+Duel | 29 | 8.639 | **0.1655** | 0.191 |
| 3 | CNN-DDQN | 29 | 8.645 | 0.1773 | 0.181 |
| 4 | CNN-DQN+MD | 29 | 8.649 | 0.1803 | 0.218 |
| 5 | MLP-DDQN | 29 | 8.652 | 0.1743 | **0.115** |
| 6 | CNN-DQN+MHA | 29 | 8.687 | 0.1856 | 0.240 |
| 7 | CNN-DQN+Duel | 29 | 8.693 | 0.1619 | 0.184 |
| 8 | CNN-DQN | 29 | 8.721 | 0.2056 | 0.183 |
| 9 | MLP-DQN | 29 | 8.778 | 0.2098 | 0.128 |
| 10 | CNN-DDQN+MHA | 29 | 8.918 | 0.2215 | 0.230 |

## 关键发现

### 1. CNN-DDQN+MD 为论文核心算法
- Quality 路径长度 Long/Short **均第一**（26.365m / 8.604m）
- SR 稳定：Long 94%（第二）、Short 92%（并列第二）
- 综合 SR + Quality 最优，路径长度为论文核心指标

### 2. DQN vs DDQN
- **Long SR**：DQN 变体整体略优（CNN-DQN 96% vs CNN-DDQN 90%）
- **Quality**：DDQN 变体路径略短（Long 26.36m vs 26.44m），更稳定

### 3. 输出头对比（MD vs Duel vs MHA）
- **MD**：Quality 路径长度最优，SR 稳定
- **Duel**：Short SR 最高（94%），曲率最低
- **MHA**：不稳定，DDQN+MHA Long SR 最低（82%）

### 4. CNN vs MLP
- **SR**：CNN 整体优于 MLP（Long 82-96% vs 88%，Short 88-94% vs 84-86%）
- **计算时间**：MLP 最快（0.11-0.26s vs CNN 0.19-0.52s）
- **Quality**：路径质量接近，但 MLP Short 路径较长

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
| delta_dot_rad_s | 转向角速度 |
| a_m_s2 | 加速度 |
| od_m | 最近障碍物距离（EDT 双圆检测） |
| collision | 是否碰撞 |
| reached | 是否到达目标 |
| stuck | 是否卡住 |
| reward | 即时奖励 |

## 数据路径

| 变体 | 目录 |
|------|------|
| CNN-DDQN | `abl_diag10k_kt02_infer_cnn_ddqn.1/` |
| CNN-DDQN+Duel | `abl_diag10k_kt02_infer_cnn_ddqn_duel.1/` |
| CNN-DDQN+MHA | `abl_diag10k_kt02_infer_cnn_ddqn_mha.1/` |
| CNN-DDQN+MD | `abl_diag10k_kt02_infer_cnn_ddqn_md.1/` |
| CNN-DQN | `abl_diag10k_kt02_infer_cnn_dqn.1/` |
| CNN-DQN+Duel | `abl_diag10k_kt02_infer_cnn_dqn_duel.1/` |
| CNN-DQN+MHA | `abl_diag10k_kt02_infer_cnn_dqn_mha.1/` |
| CNN-DQN+MD | `abl_diag10k_kt02_infer_cnn_dqn_md.1/` |
| MLP-DQN + MLP-DDQN | `abl_diag10k_kt02_infer_mlp.1/` |

## 详细日志

见 `runs/ablation_logs/ablation_20260315_diag10k_kt02.md`
