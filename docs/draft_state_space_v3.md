# 状态空间（改写 v3）

智能体在每个决策时刻需同时感知自身运动状态与环境空间结构。由于策略网络采用双分支架构分别处理向量与空间信息，本文将观测定义为标量特征向量与多通道地图张量的二元组：

$$
o_t = \bigl(\,\mathbf{f}_t,\;\mathbf{M}_t\,\bigr),\quad
\mathbf{f}_t \in \mathbb{R}^{11},\quad
\mathbf{M}_t \in \mathbb{R}^{3 \times N \times N}
$$

其中 $N$ 为地图通道的降采样边长，本文取 $N=12$。

标量特征向量 $\mathbf{f}_t$ 按功能分为三组。第一组为位姿信息，包含归一化的车辆位置 $(\tilde{x},\,\tilde{y})$、目标位置 $(\tilde{g}_x,\,\tilde{g}_y)$ 以及航向角的正弦-余弦对 $(\sin\theta,\,\cos\theta)$，其中采用三角函数编码可消除 $\theta$ 在 $\pm\pi$ 处的回绕不连续。第二组为运动状态，包含归一化的纵向速度 $\tilde{v} = v/v_{\max}$ 与前轮转角 $\tilde{\delta} = \delta/\delta_{\max}$。第三组为环境感知，包含归一化的 Dijkstra 目标距离场值 $\tilde{\phi}$，即当前位姿沿无碰撞路径到达目标的测地距离；目标相对方位角 $\tilde{\alpha} = \alpha/\pi$；以及最近障碍物距离 $\tilde{d}_{\mathrm{obs}}$，由欧氏距离变换在当前位置采样并以上限 $d_{\mathrm{cap}} = 2.0\,\mathrm{m}$ 截断后归一化。综合表示为

$$
\mathbf{f}_t = \bigl[\,
  \underbrace{\tilde{x},\, \tilde{y},\, \tilde{g}_x,\, \tilde{g}_y,\, \sin\theta,\, \cos\theta}_{\text{位姿信息}},\;
  \underbrace{\tilde{v},\, \tilde{\delta}}_{\text{运动状态}},\;
  \underbrace{\tilde{\phi},\, \tilde{\alpha},\, \tilde{d}_{\mathrm{obs}}}_{\text{环境感知}}
\,\bigr]
$$

上述所有带波浪号的分量均通过线性映射归一化至 $[-1,\,1]$，以消除量纲差异并稳定网络训练。

地图张量 $\mathbf{M}_t$ 由全局栅格地图经保持宽高比的降采样获得，沿通道维度堆叠三个语义层：

$$
\mathbf{M}_t = \bigl[\,\mathbf{O}_t;\;\mathbf{G}_t;\;\mathbf{E}_t\,\bigr]
\in \mathbb{R}^{3 \times N \times N}
$$

其中占据图 $\mathbf{O}_t \in \mathbb{R}^{N \times N}$ 以二值形式表征障碍物分布与可通行区域，采用最近邻插值下采样以保留障碍物边界的锐利特征。Dijkstra 目标距离场 $\mathbf{G}_t \in \mathbb{R}^{N \times N}$ 编码各栅格沿无碰撞路径到达目标的最短测地距离，不可达区域以有限上界填充后采用双线性插值下采样，为策略学习提供全局导航梯度，有效缓解稀疏奖励问题。欧氏距离变换图 $\mathbf{E}_t \in \mathbb{R}^{N \times N}$ 记录各栅格到最近障碍物的欧氏距离并以 $d_{\mathrm{cap}}$ 截断，同样采用双线性插值下采样，为智能体提供连续的安全裕度感知。三个通道均独立归一化至 $[-1,\,1]$。

需要指出的是，上述地图特征采用全局视角，即智能体始终观测整个环境的降采样表征而非以自身为中心的局部窗口，这使得策略网络无需显式处理坐标变换即可获取全局路径引导信息。在网络前向传播中，$\mathbf{M}_t$ 经卷积编码器提取空间特征后与 $\mathbf{f}_t$ 拼接，送入全连接层输出各动作的 $Q$ 值。
