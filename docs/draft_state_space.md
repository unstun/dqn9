# （1）状态空间 — 改写稿

智能体在每个决策时刻需要两类信息：一是自身的运动状态，如位置、速度、航向等；二是周围环境的空间结构，如障碍物分布和到目标的路径引导。据此，本文将观测分为**标量特征**和**地图特征**两部分。

## 标量特征

标量特征向量 $\mathbf{v}_{\text{scalar}} \in \mathbb{R}^{11}$ 编码车辆的运动学状态与相对目标信息：

$$
\mathbf{v}_{\text{scalar}} = \left[\,\tilde{x},\, \tilde{y},\, \tilde{g}_x,\, \tilde{g}_y,\, \sin\psi,\, \cos\psi,\, \frac{v}{v_{\max}},\, \frac{\delta}{\delta_{\max}},\, \tilde{c}_{\text{goal}},\, \tilde{\alpha},\, \tilde{d}_{\text{obs}}\,\right]
$$

各分量含义：

| 符号 | 含义 |
|------|------|
| $(\tilde{x}, \tilde{y})$ | 归一化车辆位置 |
| $(\tilde{g}_x, \tilde{g}_y)$ | 归一化目标位置 |
| $\sin\psi,\, \cos\psi$ | 航向角的三角编码 |
| $v / v_{\max}$ | 归一化纵向速度 |
| $\delta / \delta_{\max}$ | 归一化前轮转向角 |
| $\tilde{c}_{\text{goal}}$ | 归一化 Dijkstra 目标距离场值 |
| $\tilde{\alpha}$ | 目标相对方位角 |
| $\tilde{d}_{\text{obs}}$ | 归一化最近障碍物距离 |

所有分量通过线性映射归一化至 $[-1, 1]$。

## 地图特征

地图特征由全局栅格地图经保持宽高比的降采样得到，边长 $N=12$，包含三个语义通道：

| 通道 | 符号 | 作用 |
|------|------|------|
| 占据图 | $\mathbf{m}_{\text{occ}} \in \mathbb{R}^{12 \times 12}$ | 二值表征障碍物分布与可通行区域，提供局部几何约束 |
| Dijkstra 目标距离场 | $\mathbf{m}_{\text{goal}} \in \mathbb{R}^{12 \times 12}$ | 编码沿无碰撞路径到目标的最短测地距离，提供全局导航梯度，缓解稀疏奖励 |
| 欧氏距离变换图 | $\mathbf{m}_{\text{edt}} \in \mathbb{R}^{12 \times 12}$ | 记录到最近障碍物的欧氏距离，提供连续安全裕度感知 |

三个通道分别归一化至 $[-1, 1]$ 后展平。

## 完整观测

将标量特征与三通道地图特征拼接，得到完整观测：

$$
s_t = \bigl[\,\mathbf{v}_{\text{scalar}};\;\mathbf{m}_{\text{occ}};\;\mathbf{m}_{\text{goal}};\;\mathbf{m}_{\text{edt}}\,\bigr] \in \mathbb{R}^{443}
$$

总维度为 $11 + 3 \times 12^2 = 443$。
