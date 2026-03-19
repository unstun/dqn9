# 状态空间（改写 v2 — SCI 连续行文风格）

智能体在每个决策时刻需要同时感知自身运动状态与周围环境的空间结构，据此本文将观测设计为标量特征与地图特征的混合表示。标量特征向量 $\mathbf{f} \in \mathbb{R}^{11}$ 编码车辆的运动学状态与目标相对信息，定义为

$$
\mathbf{f} = \bigl[\,\tilde{x},\, \tilde{y},\, \tilde{g}^x,\, \tilde{g}^y,\, \sin\psi,\, \cos\psi,\, \tilde{v},\, \tilde{\delta},\, \tilde{\phi},\, \tilde{\alpha},\, \tilde{d}_{\min}\,\bigr]
$$

其中 $(\tilde{x}, \tilde{y})$ 与 $(\tilde{g}^x, \tilde{g}^y)$ 分别为车辆位置与目标位置，航向角 $\psi$ 采用正弦-余弦对编码以消除角度回绕引起的不连续，$\tilde{v} = v/v_{\max}$ 与 $\tilde{\delta} = \delta/\delta_{\max}$ 分别为纵向速度与前轮转向角，$\tilde{\phi}$ 为 Dijkstra 目标距离场值，即沿无碰撞路径到达目标的测地距离势函数，$\tilde{\alpha}$ 为目标相对方位角，$\tilde{d}_{\min}$ 为最近障碍物距离。上述所有带波浪号的分量均通过线性映射归一化至 $[-1,\,1]$。

地图特征将全局栅格地图经保持宽高比的降采样压缩至 $N \times N$（本文取 $N=12$），包含三个通道：占据图 $\mathbf{O} \in \mathbb{R}^{N \times N}$ 以二值形式表征障碍物分布与可通行区域；Dijkstra 目标距离场 $\mathbf{G} \in \mathbb{R}^{N \times N}$ 编码每个栅格沿无碰撞路径到达目标点的最短测地距离，为策略学习提供全局导航梯度以缓解稀疏奖励问题；欧氏距离变换图 $\mathbf{E} \in \mathbb{R}^{N \times N}$ 记录每个栅格到最近障碍物的欧氏距离，为智能体提供连续的安全裕度感知。三个通道分别归一化至 $[-1, 1]$ 后逐通道展平并与标量特征拼接，得到完整观测

$$
s_t = \bigl[\,\mathbf{f};\;\text{vec}(\mathbf{O});\;\text{vec}(\mathbf{G});\;\text{vec}(\mathbf{E})\,\bigr] \in \mathbb{R}^{11+3N^2}
$$

本文取 $N=12$，观测总维度为 $11 + 3 \times 144 = 443$。这种融合运动学标量与多通道空间地图的混合设计，使策略网络能够同时利用障碍分布、全局路径引导与安全裕度三方面信息进行决策。
