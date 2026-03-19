# 实验 2：DRL 模块消融（10 变体）

## 实验设计

- **目标**：确认最优网络架构组合
- **基底**：DQfD pretrain 40000 steps, reward_k_t=0.2, EDT diag
- **训练**：10000 episodes, seed=0
- **推理**：Seeds 42/142/242/342/442/542/642，50 runs/variant/distance
- **配置**：`configs/ablation_20260314_diag10k_kt02_*.json`

## 变体列表（10 个）

| # | 变体 | 编码器 | 目标网络 | 输出头 |
|---|------|--------|----------|--------|
| 1 | CNN-DQN | CNN | — | Linear |
| 2 | CNN-DQN+Duel | CNN | — | Dueling |
| 3 | CNN-DQN+MHA | CNN | — | MHA |
| 4 | CNN-DQN+MD | CNN | — | MD (Multi-Discrete) |
| 5 | CNN-DDQN | CNN | DDQN | Linear |
| 6 | CNN-DDQN+Duel | CNN | DDQN | Dueling |
| 7 | CNN-DDQN+MHA | CNN | DDQN | MHA |
| 8 | CNN-DDQN+MD | CNN | DDQN | MD (Multi-Discrete) |
| 9 | MLP-DQN | MLP | — | Linear |
| 10 | MLP-DDQN | MLP | DDQN | Linear |

## 核心结论

- **CNN-DDQN+MD** Quality 路径长度 Long/Short 均第一（26.364m / 8.603m），SR 稳定（94%/92%）
- MD（Multi-Discrete）全面优于 Dueling
- CNN 编码器显著优于 MLP
- DDQN 目标网络提升稳定性

## 详细日志

见 `runs/ablation_logs/ablation_20260315_diag10k_kt02.md`

## 数据路径

- `abl_diag10k_kt02_infer_cnn_ddqn/` — CNN-DDQN 推理结果
- `abl_diag10k_kt02_infer_cnn_ddqn_duel/` — CNN-DDQN+Duel
- `abl_diag10k_kt02_infer_cnn_ddqn_md/` — CNN-DDQN+MD
- `abl_diag10k_kt02_infer_cnn_ddqn_mha/` — CNN-DDQN+MHA
- `abl_diag10k_kt02_infer_cnn_dqn/` — CNN-DQN
- `abl_diag10k_kt02_infer_cnn_dqn_duel/` — CNN-DQN+Duel
- `abl_diag10k_kt02_infer_cnn_dqn_md/` — CNN-DQN+MD
- `abl_diag10k_kt02_infer_cnn_dqn_mha/` — CNN-DQN+MHA
- `abl_diag10k_kt02_infer_mlp/` — MLP-DQN + MLP-DDQN
