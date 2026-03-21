# CNN-MD-DDQN 架构图参考素材

> 收集日期：2026-03-20
> 用途：为论文 Section 3.4.4 的网络架构图提供视觉参考

## 按组件分类

### Dueling DQN 双流结构
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `02_wang2016_dueling_dqn.png` | Wang et al. 2016 (ICML) — Dueling DQN 原文 Figure 1 | 经典的单流 vs 双流（Value + Advantage）分叉对比图；3D 卷积块画法 |
| `11_pmr_dueling_dqn.png` | PMR-Dueling DQN (MDPI Sensors 2024) | Input → Conv → Flatten → Value/Advantage 菱形分叉 → 聚合 → Output；不同形状区分不同层类型 |
| `08_niroui2020_fully_conv_qnet.png` | Niroui et al. 2020 — FCQN | 全卷积 Dueling 变体：1x1 conv (Advantage) / Global Maxpool (State value)；颜色区分主网络和辅助任务 |

### CNN 双输入（栅格地图 + 标量拼接）
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `03_xu2022_three_cnn_archs.png` | Xu et al. 2022 (arXiv:2208.08034) | **最相关**：三种 CNN 架构对比（FC / 1D-CNN / 2D-CNN）；occupancy grid → CNN → Flatten → Concatenate with 标量 → FC → action；绿色=输入，蓝色=网络层，粉色=输出 |

### Multi-Head Attention 模块
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `04a_transformer_full_architecture.png` | Vaswani et al. 2017 — "Attention Is All You Need" | Transformer 完整架构；MHA + Add & Norm 残差连接的标准画法 |
| `04b_transformer_scaled_dot_product_attention.png` | 同上 Figure 2 左 | Q/K/V → MatMul → Scale → Mask → SoftMax → MatMul 的详细流程 |
| `04c_transformer_multi_head_attention.png` | 同上 Figure 2 右 | h 个并行 head → Concat → Linear 的宏观画法；可直接参考画 SpatialMHA(h=4) |
| `07a_alammar_mha_qkv.png` | Jay Alammar Illustrated Transformer | 教科书级图示：两个 attention head 的 Q/K/V 权重矩阵展示；颜色编码极佳 |
| `07b_alammar_mha_concat.png` | 同上 | Z0-Z7 Concatenate → W^O → Z 的步骤分解 |
| `07c_alammar_mha_complete.png` | 同上 | MHA 完整 5 步流程：Input → Embed → Split into h heads → Attention → Concat → W^O |
| `10b_mha_with_masking.png` | boring-guy.sh 博客 | Self Attention 详细流程 + Multi Head Attention 宏观视图并排；带 Mask 集成 |
| `13b_mha_dqn_uav_architecture.png` | MHA-DQN UAV (MDPI Electronics 2025) | **最相关**：State → FC → 4 Attention Head → Concat+Projection → Dueling (Value+Advantage) → Q → Action Selection；同时展示训练循环 |

### Spatial/Channel Attention（CNN 特征图注意力）
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `05a_cbam_overview.png` | CBAM (ECCV 2018) | 3D 立方体表示特征图 + Channel Att ⊗ Spatial Att 串联结构 |
| `05b_cbam_channel_attention.png` | 同上 | Channel + Spatial Attention 子模块详图；3D 特征图维度变化清晰 |

### CNN 骨干可视化
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `06a_darqn_architecture.png` | DARQN (2015) | CNN → Attention(g) → LSTM → Q(s,a) 的时序展开；展示 attention 如何桥接 CNN |
| `06b_darqn_cnn.png` | 同上 | 经典 3D 透视 CNN 可视化：84×84 → 20×20×32 → 9×9×64 → 7×7×256；维度标注清晰 |

### Action Masking
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `10a_action_masking_logits.png` | boring-guy.sh 博客 | State → NN → Action logits → Mask → Masked Q values (无效→-∞)；简洁直观 |

### DQN/DDQN 系统级训练架构
| 文件 | 来源 | 可借鉴之处 |
|------|------|-----------|
| `09a_frontiers2025_dqn.png` | Frontiers in Neurorobotics 2025 | DQN 完整训练流程：Environment ↔ Current/Target Network + Experience Replay + Loss |
| `09b_frontiers2025_ddqn.png` | 同上 | DDQN：Current Network 选动作、Target Network 评估的分离机制 |

---

## 对 CNN-MD-DDQN 架构图的绘制建议

### 你的架构关键组件
```
[2-ch Map (Occ+GoalDist)] → Conv(2→32,k3s1) → Conv(32→64,k3s2) → Conv(64→64,k3s2)
                            → SpatialMHA(h=4) → Flatten(576)
[11-d Scalar] ──────────────────────────────────────────────→ Concat(587)
                                                               → FC(587→256)+ReLU (Shared)
                                                               ├→ Value: FC(256→256→1)
                                                               └→ Advantage: FC(256→256→35)
                                                               → Q = V + A - mean(A)
                                                               → Action Mask → a*
```

### 推荐参考组合
1. **CNN 骨干** → 参考 `06b_darqn_cnn.png` 的 3D 透视立方体
2. **双输入拼接** → 参考 `03_xu2022_three_cnn_archs.png` 的绿/蓝/粉色编码
3. **SpatialMHA 模块** → 参考 `04c_transformer_multi_head_attention.png` 的结构 + `05b_cbam` 的 3D 特征图
4. **Dueling 双流** → 参考 `02_wang2016_dueling_dqn.png` 的经典分叉 + `13b` 的彩色方块
5. **Action Mask** → 参考 `10a_action_masking_logits.png` 的 logit-level masking
6. **整体风格** → 参考 `13b_mha_dqn_uav_architecture.png` 的彩色模块化流程图

### 绘图工具
- **TikZ** (当前使用): `paper/figures/fig_architecture.tex`
- **NNTikZ**: github.com/fraserlove/nntikz — 有 MHA TikZ 模板
- **PlotNeuralNet**: github.com/HarisIqbal88/PlotNeuralNet — 3D CNN 块
- **draw.io**: 快速迭代原型 → 导出 SVG → Inkscape 微调
