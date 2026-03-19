# DeepMNavigate: 基于深度强化学习的多机器人导航——统一局部与全局避碰

**作者：** Qingyang Tan¹, Tingxiang Fan², Jia Pan², Dinesh Manocha¹

视频链接：https://youtu.be/LWLBxWuwPeU

---

## 摘要

本文提出一种新颖的算法 DeepMNavigate，利用深度强化学习（Deep Reinforcement Learning, DRL）实现密集场景下的全局多智能体导航（multi-agent navigation）。该方法通过运动信息地图（motion information maps）为每个机器人提供局部和全局信息。我们使用三层卷积神经网络（Convolutional Neural Network, CNN）将这些地图作为输入，生成合适的动作以驱动每个机器人到达其目标位置。该方法具有通用性，通过多场景、多阶段训练算法学习最优策略，并可直接处理原始传感器测量数据作为局部观测。我们在包含窄通道和数十个智能体的密集复杂基准测试中展示了算法性能，并突出了该算法相较于先前学习方法和几何分散式算法在复杂场景中的优势。

---

## I. 引言

多机器人系统（multi-robot systems）正越来越多地应用于不同领域，包括监控、质量控制系统、自主导引车辆、仓储、清洁机器等。一个关键挑战是开发高效的算法，使这些机器人能在复杂场景中导航，同时避免彼此碰撞以及与环境中障碍物的碰撞。随着使用的机器人数量增多，需要更高效的方法来处理密集和复杂的场景。

多智能体导航在机器人学、人工智能和计算机动画领域已被广泛研究。从宏观层面看，以往的方法可分为集中式规划器（centralized planners）[1]–[6] 和分散式规划器（decentralized planners）[7]–[10]。分散式方法的一个优点是能扩展到大量智能体，但难以对生成的轨迹提供任何全局保证 [11]，也难以处理包含窄通道的挑战性场景（如图 1 和图 2 所示）。

近年来，有大量工作致力于开发基于学习的新规划算法 [12]–[17]，用于在密集场景中导航一个或多个机器人。这些学习方法大多通过多场景、多阶段训练算法来学习最优策略。然而，当前基于学习的方法仅限于使用局部信息，而未利用任何全局信息。因此，在密集环境或窄通道中使用这些方法存在困难。

**主要成果：** 我们提出一种基于强化学习的新颖多机器人导航算法 DeepMNavigate，它利用局部和全局信息的组合。我们使用多阶段训练方案，该方案利用包含全局信息的多种多机器人仿真场景。

> **图 1：** 圆形交叉（Circle Crossing）场景——90 个机器人在圆形交叉场景中的仿真轨迹。黄色点为机器人的初始位置，红色点为直径对面的目标位置。DeepMNavigate 算法能够在轨迹上无碰撞地处理此类场景，所有机器人均到达目标。仅使用局部方法的先前学习方法 [16]、[12]、[14] 无法处理此类场景，机器人会陷入停滞。

在训练方面，我们使用运动信息地图来表示全局信息，该地图包含每个智能体或机器人的位置。我们将机器人信息放置在相应位置以生成位图，并将结果地图作为三层 CNN 的输入。我们的 CNN 结合此全局信息和场景中的局部观测来生成合适的动作，驱动每个机器人无碰撞地到达目标。我们在包含数十个机器人（例如 90 个）在窄通道紧凑场景中导航的密集环境中评估了该算法。与先前的多机器人方法相比，我们的方法具有以下优势：

1. 我们在网络中使用运动信息地图形式的全局知识来提升 DRL 的性能，这也带来了更高的奖励值。
2. 该方法可随机器人数量扩展，能计算无碰撞且平滑的轨迹。在配备 32 核 CPU 和一块 NVIDIA RTX 2080 Ti 的 PC 上，运行训练好的系统在 10–90 个机器人的多机器人系统上仅需数十秒。
3. 我们可以轻松处理具有挑战性的多机器人场景，如机器人位置互换或多个窄走廊，这对于先前的几何分散式方法或局部学习方法而言十分困难。特别是，我们在五个困难环境中展示了性能表现，这些环境与训练场景截然不同且拥有更多智能体，充分证明了方法的泛化能力。

---

## II. 相关工作

### A. 几何多机器人导航算法

大多数先前的算法基于几何技术，如基于采样的方法、几何优化或完备运动规划算法。集中式方法假设每个机器人可以基于某些全局数据结构或通信系统获取其他机器人的完整状态信息 [18]–[21]，并计算安全、最优且完备的导航解。然而，它们无法扩展到拥有数十个机器人的大型多机器人系统。许多实用的几何分散式多智能体方法基于互惠速度障碍（Reciprocal Velocity Obstacles）[7] 或其变体 [22]。这些合成方法可在学习算法的训练阶段使用。

### B. 基于学习的导航方法

基于学习的碰撞避免技术通常尝试使用从不同任务收集的数据来优化参数化策略。许多导航算法采用监督学习（supervised learning）范式来训练碰撞避免策略。Muller 等人 [23] 提出了一种基于视觉的静态障碍物避让系统，使用六层 CNN 将输入图像映射到转向角。Zhang 等人 [24] 描述了一种基于后继特征（successor features）的深度强化学习算法，用于基于原始传感数据的机器人导航任务。Barreto 等人 [25] 将迁移学习（transfer learning）应用于为新问题实例部署策略。Sergeant 等人 [26] 提出了一种基于多模态深度自编码器（multimodal deep autoencoders）的方法，使机器人能够通过观察人类远程操控时收集的传感器输入和运动命令数据集来学习导航。Ross 等人 [27] 采用模仿学习（imitation learning）技术，基于人类飞行员的知识训练反应式航向策略。Pfeiffer 等人 [28] 利用专家示范将激光扫描和目标位置映射到运动命令。这些方法需要在不同环境中收集训练数据，性能受限于训练集的质量。

为克服监督学习的局限性，Tai 等人 [29] 提出了一种无地图运动规划器（mapless motion planner），端到端训练，无需任何人工设计特征或先验演示。Kahn 等人 [30] 提出了一种不确定性感知的基于模型的学习算法，估计碰撞概率，然后利用该信息在训练时最小化碰撞。为将基于学习的方法扩展到高度动态环境，提出了一些分散式技术。Godoy 等人 [31] 提出了一种贝叶斯推理方法，计算在驱动机器人到达目标的同时最小化碰撞次数的规划。Chen 等人 [14]、[32] 和 Everett 等人 [15] 提出了基于深度强化学习的多机器人碰撞避免策略，需要部署多个传感器来估计附近智能体和移动障碍物的状态。Yoon 等人 [33] 将集中训练与分散执行的框架扩展为对智能体间通信进行额外优化。Fan 等人 [12] 和 Long 等人 [16]、[17] 描述了一种分散式多机器人碰撞避免框架，其中每个机器人独立做出导航决策，无需与其他智能体通信。该框架已在多传感器和显式行人运动预测方面得到扩展 [34]。其他方法考虑了社会约束 [15]。然而，所有这些方法都未利用关于机器人或环境的全局信息，而这些信息可用于提高生成路径的最优性或处理具有挑战性的窄通道场景。

> **图 2：** 窄走廊（Narrow Corridor）场景——两组机器人（共 20 个）通过窄走廊交换位置。(a) 和 (c) 中黄色点为初始位置，红色点为最终位置。(b) 和 (d) 用颜色和透明度突出轨迹上的时间信息。(a)(b) 为 DeepMNavigate 的结果，能计算 20 个机器人的无碰撞轨迹；(c)(d) 为局部学习方法 [16] 的结果，智能体陷入停滞。先前的局部规划方法 [16] 仅能处理最多 12 个智能体的此类场景，几何分散式方法 [7] 无法处理此类情况。

---

## III. 多机器人导航

### A. 问题描述与符号

我们考虑非完整约束差速驱动（non-holonomic differential drive）机器人的多机器人导航问题。目标是设计一种方案，在避免与障碍物和其他机器人碰撞的同时，在密集和一般环境中良好工作。我们针对二维情况进行描述，但可扩展到三维工作空间和具有其他动力学约束的机器人。

设机器人数量为 $N_{rob}$。将每个机器人表示为半径为 $R$ 的圆盘。在每个时间步 $t$，第 $i$ 个机器人（$1 \leq i \leq N_{rob}$）可获取观测 $o^t_i$，然后计算动作 $a^t_i$，驱动第 $i$ 个机器人从当前位置 $p^t_i$ 到达目标 $g^t_i$。每个机器人的观测包含四部分：$o^t = [o^t_z, o^t_g, o^t_v, o^t_M]$，其中 $o^t_z$ 为传感器测量值（如激光传感器），$o^t_g$ 为相对目标位置，$o^t_v$ 为当前速度，$o^t_M$ 为机器人运动信息（包含系统的全局状态，详见第 IV 节）。本文重点分析并将运动信息纳入导航系统。同时，环境中有 $N_{obs}$ 个静态障碍物，用 $B_k$ 表示第 $k$ 个静态障碍物所占区域。计算出的动作 $a^t$ 在时间步 $\Delta t$ 内驱动机器人到达目标，同时避免与其他机器人和障碍物碰撞，直到接收到下一个观测 $o^{t+1}$。

设 $L$ 为所有机器人的轨迹集合，满足机器人运动学约束：

$$L = \{l_i, i = 1, ..., N_{rob} | v^t_i \sim \pi_\theta(a^t_i | o^t_i), p^t_i = p^{t-1}_i + \Delta t \cdot v^t_i, \forall j \in [1, N_{rob}], j \neq i, \|p^t_i - p^t_j\| > 2R \wedge \forall k \in [1, N_{obs}], \forall q \in B_k, \|p^t_i - q\| > R \wedge \|v^t_i\| \leq v^{max}_i\}$$
(1)

其中 $p^0_i$ 为机器人的初始位置，$p^t_i$ 为时间步 $t$ 的位置。$v^t_i$ 为二维平面上的当前线速度（即 $v_x, v_y$），由动作 $a^t_i$ 产生。我们仿真的智能体为非完整约束，只能控制 X 轴上的线速度和 Z 轴上的角速度（用于描述二维平面上的旋转）。

### B. 基于强化学习的多智能体导航

本方法建立在使用各种局部信息观测的先前强化学习方法之上。其中一些仅使用第 III-A 节中提到的四个要素中的三个。$o^t_z$ 可能包含传感器最近三个连续帧的测量值。这些情况下的相对目标位置 $o^t_g$ 是一个二维向量，表示以机器人当前位置为参考的极坐标目标位置。观测速度 $o^t_v$ 包含机器人的当前速度。这些观测通过训练期间聚合的统计量进行归一化 [35]、[36]。这种归一化可使强化学习训练更稳定并提升性能。差速机器人的动作包括平移和旋转速度，即 $a^t = [v^t, \omega^t]$，$v \in (0, 1)$，$\omega \in (-1, 1)$。我们使用以下奖励函数引导机器人团队：

$$r^t_i = (gr)^t_i + (cr)^t_i + (\omega r)^t_i$$
(2)

当机器人接近或到达目标时，获得如下奖励：

$$(gr)^t_i = \begin{cases} r_{arrival} & \text{if } \|p^t_i - g_i\| < 0.1 \\ r_{approaching}(\|p^{t-1}_i - g_i\| - \|p^t_i - g_i\|) & \text{otherwise} \end{cases}$$
(3)

$\|p_i - g_i\|$ 表示机器人与目标之间的距离。发生碰撞时，使用以下函数进行惩罚：

$$(cr)^t_i = \begin{cases} r_{collision} & \text{if } \|p^t_i - p^t_j\| < 2R \text{ or } \|p^t_i - q\| < R, q \in B_k \\ 0 & \text{otherwise} \end{cases}$$
(4)

除碰撞避免外，另一目标是生成平滑路径。一种简单技术是在旋转速度较大时施加惩罚。虽然这不是获得平滑路径的标准方法，但我们发现该技术在实践中能有效实现平滑轨迹：

$$(\omega r)^t_i = \begin{cases} r_{smooth}|\omega^t_i| & \text{if } |\omega^t_i| > 0.7 \\ 0 & \text{otherwise} \end{cases}$$
(5)

其中 $r_{arrival}$、$r_{approaching}$、$r_{collision}$ 和 $r_{smooth}$ 为控制奖励的参数。这些参数为智能体提供奖励反馈，使训练过程更加稳定 [37]。实践中，可调整奖励参数以获得期望行为（例如通过更大的碰撞惩罚来学习更保守的行为）。在不同环境中使用本方法时，我们不改变奖励函数。

> **图 3：** 策略网络（DeepMNavigate）架构——包括全局地图和局部地图。全局地图基于世界坐标系，每个局部地图以对应机器人的当前位置为中心。红色机器人代表地图对应的机器人，黑色机器人代表邻近机器人，黄色星号代表目标，蓝色区域为障碍物。实现中地图被离散化并赋予不同值。使用二维 CNN 处理来自地图的额外全局信息输入，使用全连接网络为每个机器人计算动作。

---

## IV. DeepMNavigate: 利用全局信息的轨迹计算

本节介绍我们新颖的基于学习的多智能体导航算法，该算法利用其他智能体的位置信息。公式基于运动信息地图，并使用三层 CNN 为每个智能体生成合适的动作。

### A. 运动信息地图

先前基于规则的分散式方法如 [7] 使用每个智能体的位置和速度信息来计算局部最优且无碰撞的轨迹。我们的目标是计算类似的状态信息，以设计更好的基于学习的导航算法。这种状态信息可以通过与附近机器人的某种通信获取，或使用深度网络从原始传感器数据计算得到。在我们的公式中，我们使用包含每个智能体位置的地图作为输入。具体而言，我们使用两种不同的地图表示：一种对应于基于世界坐标系的所有机器人，称为全局地图（global-map）；另一种以每个机器人的当前位置为中心，使用相对坐标系，称为局部地图（local-map）。

我们使用以下方法计算全局地图和局部地图。在每个时间步 $t$，我们指定第 $i$ 个机器人在世界坐标系中的位置为 $x^t_i \in \mathbb{R}^2$。我们还使用目标位置 $g_i$（$\forall 1 \leq i \leq N_{rob}$）和障碍物信息 $B_k$（$\forall 1 \leq k \leq N_{obs}$）来为第 $i$ 个机器人构建地图 $M^t_i \in \mathbb{R}^{h \times w}$。假设仿真机器人场景大小为 $H \times W$，其中 $H$ 为高度，$W$ 为宽度，世界坐标系原点位于 $(\frac{H}{2}, \frac{W}{2})$。地图的每个像素 $M^t_i(p, q)$（$\forall 1 \leq p \leq h, 1 \leq q \leq w$）指示在世界坐标系的小区域 $A_{pq}$ 中存在何种类型的物体。假设每个物体的半径为 $r_i$，则 $M^t_i(p, q)$ 定义为：

$$M^t_i(p, q) = \begin{cases} 1, & \{y | \|y - x^t_i\| \leq R\} \cap A_{pq} \neq \emptyset \\ 2, & \exists 1 \leq j \leq N_{rob}, j \neq i, \text{ s.t. } \{y | \|y - x^t_j\| \leq R\} \cap A_{pq} \neq \emptyset \\ 3, & \{y | \|y - g_i\| \leq R\} \cap A_{pq} \neq \emptyset \\ 4, & \exists 1 \leq k \leq N_{obs}, \text{ s.t. } B_k \cap A_{pq} \neq \emptyset \\ 0, & \text{otherwise} \end{cases}$$
(6)

其中"1"代表对应机器人，"2"代表邻近机器人，"3"代表机器人目标，"4"代表障碍物，"0"代表空白背景（即自由空间）。如图 3 所示。

在某些场景中，机器人的运动可能受到静态障碍物或不可达区域的限制。我们的全局地图计算通过 $M^t_i(p, q)$ 的表示来考虑这一点。然而，这些地图可能无法捕获没有明确边界或面积非常大的场景。如果对所有智能体使用世界坐标表示的全局地图，生成的地图将极其庞大，导致高计算成本和内存开销。在这些情况下，我们对每个智能体使用局部地图，而非考虑大小为 $H \times W$ 的整个场景。这些局部地图仅考虑固定大小 $H_l \times W_l$ 的相对较小邻域内的信息。局部邻域大小 $(H_l, W_l)$ 可以调整以获得不同应用的更好性能。除位置信息外，这些地图还可以包含机器人的其他状态信息，如速度、朝向或动力学约束等附加通道。

### B. 近端策略优化

我们使用近端策略优化（Proximal Policy Optimization, PPO）[38] 来优化整个系统。该训练算法具有信赖域方法的稳定性和可靠性，即尝试在每一步计算一个更新，在最小化代价函数的同时确保与先前策略的偏差相对较小。由此产生的近端策略算法在若干连续仿真（即整个系统中每个机器人到达目标或因碰撞停止运行）之后，使用所有步骤来更新网络，而非仅使用一步来确保网络优化的稳定性。在这种情况下，如果按照公式 (6) 的形式将机器人位置或运动信息存储为密集矩阵，将需要大量内存并增加总体训练时间。因此，我们对 $M^t_i$ 使用稀疏矩阵（sparse matrix）表示。我们基于每个机器人的当前位置、目标位置和障碍物信息，使用公式 (6) 计算 $M^t_i$ 的非零项。为将输入馈送到神经网络，我们使用临时稀疏存储生成密集表示。这一设计选择使我们能够使用仅 2.5GB 内存训练包含 58 个智能体执行 450 个动作的轨迹系统。关于训练步骤的更多细节见第 V 节。

### C. 网络

为分析大型矩阵并为输入 $M^t_i$ 生成低维特征，我们使用卷积神经网络，因为此类网络结构适合处理图像类输入（或我们的地图表示）。网络包含三个卷积层，架构如图 3 所示。网络提取环境中不同物体的相关位置，并能引导整体规划器避开其他智能体和障碍物以到达目标。

处理原始传感器数据（如二维激光扫描数据）的方法使用与局部方法 [16] 相同的结构，即两层一维卷积网络。总体而言，我们使用两层全连接网络，以观测特征作为输入，包括一维和二维 CNN 生成的特征、相关目标位置和观测速度，生成每个机器人的动作输出和局部路径。整体网络管线如图 3 和算法 1 所示。

**算法 1：DeepMNavigate 策略制定**

```
1: for 时间步 t = 1, 2, ... do
2:     // 每个机器人并行独立运行
3:     for 机器人 i = 1, 2, ...N do
4:         收集观测 o^t_i = [o^t_z, o^t_g, o^t_v, o^t_M]
5:         运行策略网络 π_θ 表示的策略，获取动作 a^t_i
6:         根据动作 a^t_i 更新机器人位置 p^t_i
7:     end for
8: end for
```

### D. 网络训练

我们的训练策略扩展了基于局部信息的学习算法 [16]、[14] 所使用的方法。为加速训练过程，我们将整体训练计算分为两个阶段。第一阶段，我们在完全自由环境中使用 $k$ 个机器人（例如 $k=20$），随机初始位置和随机目标。第二阶段，我们加入更具挑战性的环境，如窄通道、随机障碍物等。训练场景如图 5 所示。这些多样的训练环境和系统中大量机器人可以产生良好的整体策略。

> **表 I：** 处理全局信息的卷积神经网络架构和超参数

| 层 | 卷积滤波器 | 步长 | 填充 | 激活函数 | 输出尺寸 |
|---|---|---|---|---|---|
| 输入 | - | - | - | - | 250×250×1 |
| Conv 1 | 7×7×1×8 | 1×1 | SAME | ReLU | 250×250×8 |
| Max Pooling 1 | 3×3 | 2×2 | SAME | - | 125×125×8 |
| Conv 2 | 7×7×8×12 | 1×1 | SAME | ReLU | 125×125×12 |
| Max Pooling 2 | 3×3 | 2×2 | SAME | - | 63×63×12 |
| Conv 3 | 7×7×12×20 | 1×1 | SAME | ReLU | 63×63×20 |
| Max Pooling 3 | 3×3 | 2×2 | SAME | - | 32×32×20 |
| 展平 | - | - | - | - | 20480 |
| 全连接 | 20480×128 | - | - | ReLU | 384 |
| 全连接 | 128×64 | - | - | ReLU | 256 |

此外，使用全局信息会导致更大的网络参数。我们使用 20480×128 的全连接层，这比仅考虑局部信息的相对较小的简单网络更难训练。为加速训练过程并生成准确结果，我们不从头训练整个网络，而是包含预训练（pre-training）。在第一阶段，我们使用预训练的局部信息网络部分，重新训练带有全局信息（即全局地图）附加结构的网络。此预训练阶段使用 [16] 中提出的参数。图 4 展示了第二阶段训练期间奖励随迭代次数变化的情况，并与局部方法 [16] 的整体奖励计算进行了比较。

> **图 4：** 训练第二阶段奖励与迭代次数的关系。与仅使用局部信息的强化学习算法 [16] 相比，使用局部和全局信息的本方法获得了更高的奖励。

需注意，由于我们使用了二维卷积神经网络，整体训练算法在每次迭代中需要更多时间（本方法约 1200 秒 vs [16] 约 400 秒）。因此，如图 4 所示，我们没有执行与 [16] 相同的训练迭代次数。总训练时间约为 40 小时。

> **图 5：** 训练第二阶段使用的挑战性环境对应的场景（共 7 个场景）。这种两阶段训练提高了性能。

---

## V. 实现与性能

本节讨论多智能体导航算法 DeepMNavigate 在复杂场景中的性能，并突出其相对于仅使用局部信息的先前强化学习方法 [16] 的优势。

### A. 参数

仿真中，半径设为 $R = 0.12$ m。当前实现中，全局地图设 $H = W = 500$ m，每个局部地图设 $H_l = W_l = 250$ m。两种情况下均设 $w = h = 250$。虽然更大的地图（如 $w = h = 500$）可包含系统中更多细节，但会显著增加网络大小和最终运行时间。例如，CNN 的内存需求随输入大小呈二次方增长。当前实现使用配备 32 核 CPU、32GB 内存和一块 NVIDIA RTX 2080 Ti 的 PC。

引入额外全局信息后，算法消耗 1.63GB CPU 内存、970MB GPU 内存，每个时间步计算需 0.25 秒；相比之下 [16] 消耗 1.57GB CPU 内存、340MB GPU 内存和 0.2 秒。额外开销并不显著。

奖励函数参数设置为：$r_{arrival} = 15$，$r_{collision} = -15$，$r_{approaching} = 2.5$，$r_{smooth} = -0.1$。选择 $r_{arrival}$ 和 $r_{collision}$ 等量级以获得有效的安全行为，这是评估机器人轨迹的关键指标；$r_{approaching}$ 用于鼓励机器人尽快接近目标，提供密集反馈以加速收敛；较小的 $r_{smooth}$ 值用于正则化轨迹使其更平滑。

> **图 6：** 带障碍物的房间（Room with Obstacles）——20 个机器人在包含多个障碍物的房间中的轨迹。黄色为初始位置，红色为最终位置。仅使用局部方法的先前学习方法 [16]、[14] 在障碍物数量或智能体数量增加时需要更多时间且可能无法处理此类场景。

> **图 7：** 随机起点和目标（Random Start and Goal Positions）——20 个机器人从随机位置出发到达随机目标位置的仿真轨迹，场景中包含随机位置和方向的障碍物。黄色点为初始位置，红色点为最终位置，蓝色区域为障碍物。

> **图 8：** 房间疏散（Room Evacuation）——6 个机器人使用本算法疏散房间的仿真轨迹。黄色为初始位置，红色为最终位置。即使存在窄通道，DeepMNavigate 算法仍能良好工作。此基准测试与训练数据差异很大。

### B. 评估指标与基准测试

为评估导航算法的性能，我们使用以下指标：

- **成功率（Success Rate）：** 在一定时间限制内无碰撞到达目标的机器人数量与环境中机器人总数之比。
- **碰撞或停滞率（Collision or Stuck Rate）：** 若机器人无法在有限时间内到达目的地或发生碰撞，则分别视为停滞或碰撞。
- **额外时间（Extra Time）：** 所有机器人的平均行程时间与机器人行程时间下界之差。后者计算为在不检查任何碰撞的情况下以最大速度直线行驶到目标的平均行程时间。
- **平均速度（Average Speed）：** 导航期间所有机器人的平均速度。

我们在五个具有挑战性和代表性的基准测试中评估了算法：

- **圆形交叉（Circle Crossing）：** 机器人均匀分布在圆周上，每个机器人的目标位置在圆的直径对面。该场景广泛用于先前的多智能体导航算法 [7]、[16]（图 1）。
- **窄走廊（Narrow Corridor）：** 两组机器人通过窄走廊交换位置。此基准测试对几何分散式方法 [7] 具有挑战性，它们无法使机器人通过窄通道导航（图 2）。
- **带障碍物的房间（Room with Obstacles）：** 机器人穿过满是障碍物的房间，从一侧到另一侧。仅使用局部信息的方法 [16] 需要更多时间寻找到目标的路径，甚至可能因缺少全局信息而失败（图 6）。
- **随机起点和目标（Random Starts and Goals）：** 机器人从随机初始位置出发移动到随机目标位置，同时存在随机位置和方向的障碍物。全局信息有助于找到更安全、更快的路径（图 7）。
- **房间疏散（Room Evacuation）：** 机器人从随机初始位置出发疏散到房间外部，需要通过一扇小门同时避免碰撞（图 8）。

### C. 全新且不同的基准测试

我们在与训练数据在布局和窄通道包含方面差异很大的基准测试中评估了方法性能。这些基准测试还使用了不同数量的智能体。在圆形交叉基准测试中，我们仅用 12 个智能体训练，但在类似场景中用 90 个智能体进行评估。此外，窄走廊和房间疏散等基准测试与训练数据集差异很大。DeepMNavigate 仍能为所有智能体计算无碰撞的平滑轨迹，且每个智能体都能到达目标位置。如图 2 所示，局部学习方法 [16] 不考虑全局地图信息，在此类场景中会失败。相比之下，我们的方法使机器人能根据全局地图信息学习互惠导航行为，无需任何动作决策上的通信。

> **图 9：** 使用方法 [39] 在窄走廊基准测试中生成的扰动显著性图（perturbation saliency）。红色点代表当前智能体，黄色点代表周围智能体。黄色区域为动作策略的显著性区域。

### D. 定量评估

我们根据上述不同评估指标评估了算法性能。在圆形交叉场景中，[16] 的失败率随机器人数量或密度的增加而上升。然而，我们的方法始终表现稳定，能够避免死锁情况。有时某些机器人可能需要走更长的路径以避免拥堵，这可能降低机器人的效率。总体而言，在圆形交叉等高密度基准测试中，我们获得了优于 [16] 或分散式碰撞避免方法的性能。

> **表 II：** 本方法（DeepMNavigate）与基于局部信息的先前方法在不同基准测试（上：圆形交叉；左下：窄走廊；右下：带障碍物的房间）上使用不同数量智能体的性能对比。

**圆形交叉（Circle Crossing）：**

| 指标 | 方法 | 30 (8m) | 40 (8m) | 50 (8m) | 60 (8m) | 70 (8m) | 80 (12m) | 90 (12m) |
|---|---|---|---|---|---|---|---|---|
| 成功率 | [16] | 1 | 0.975 | 0.96 | 0.95 | 0.929 | 0.7375 | 0.722 |
| | **本方法** | **1** | **1** | **1** | **1** | **0.986** | **1** | **1** |
| 停滞/碰撞率 | [16] | 0/0 | 0/0.025 | 0/0.04 | 0/0.05 | 0/0.071 | 0.175/0.0875 | 0.233/0.044 |
| | **本方法** | **0/0** | **0/0** | **0/0** | **0/0** | **0/0.014** | **0/0** | **0/0** |
| 额外时间 | [16] | 4.32 | 8.20 | 7.86 | 11.31 | 13.54 | 15.12 | 15.33 |
| | 本方法 | 8.95 | 8.73 | 9.74 | 10.10 | 12.47 | 17.55 | 34.13 |
| 平均速度 | [16] | 0.787 | 0.661 | 0.671 | 0.586 | 0.542 | 0.543 | 0.610 |
| | 本方法 | 0.641 | 0.647 | 0.622 | 0.613 | 0.562 | 0.506 | 0.413 |

**窄走廊（Narrow Corridor）：**

| 指标 | 方法 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| 成功率 | [16] | 0.875 | 0.75 | 0.5625 | 0.0 |
| | **本方法** | **1** | **1** | **1** | **1** |
| 停滞/碰撞率 | [16] | 0/0.125 | 0/0.25 | 0.3125/0.125 | 1/0 |
| | **本方法** | **0/0** | **0/0** | **0/0** | **0/0** |
| 额外时间 | [16] | 4.8 | 12.51 | 30.72 | - |
| | **本方法** | **2.7** | **4.08** | **5.33** | **8.36** |
| 平均速度 | [16] | 0.654 | 0.411 | 0.231 | - |
| | **本方法** | **0.742** | **0.657** | **0.595** | **0.483** |

**带障碍物的房间（Room with Obstacles）：**

| 指标 | 方法 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| 成功率 | [16] | 1 | 0.7 | 0.6 | 0.7 |
| | **本方法** | **1** | **1** | **1** | **1** |
| 停滞/碰撞率 | [16] | 0/0 | 0.1/0.2 | 0.133/0.267 | 0.15/0.15 |
| | **本方法** | **0/0** | **0/0** | **0/0** | **0/0** |
| 额外时间 | [16] | 9.56 | 5.13 | 6.38 | 10.00 |
| | **本方法** | **2.19** | **2.55** | **3.47** | **7.80** |
| 平均速度 | [16] | 0.707 | 0.734 | 0.722 | 0.508 |
| | **本方法** | **0.859** | **0.828** | **0.776** | **0.582** |

与圆形交叉场景不同，其他基准测试在环境中加入了一些静态障碍物。在这种情况下，我们的方法将地图信息集成到策略网络中，并利用该信息处理此类静态障碍物和窄通道。实验结果表明，我们的方法在成功率和效率指标方面均优于 [16]。此外，即使机器人密度增加，我们的方法仍表现稳健。

> **表 III：** 本方法与先前学习算法在随机起点和目标基准测试中的性能对比。

| 指标 | 方法 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| 成功率 | [16] | 1 | 0.867 | 0.825 | 0.76 |
| | **本方法** | **1** | **1** | **1** | **1** |
| 停滞/碰撞率 | [16] | 0/0 | 0.033/0.1 | 0.05/0.125 | 0.2/0.04 |
| | **本方法** | **0/0** | **0/0** | **0/0** | **0/0** |
| 额外时间 | [16] | 4.60 | 9.39 | 13.83 | 12.69 |
| | **本方法** | **3.60** | **4.06** | **10.48** | **11.29** |
| 平均速度 | [16] | 0.601 | 0.432 | 0.355 | 0.332 |
| | **本方法** | **0.675** | **0.635** | **0.404** | **0.383** |

评估多机器人系统性能的另一个重要标准是停滞或碰撞率，衡量无法到达目标或在途中碰撞的机器人数量。如表 II 和表 III 所示，我们在所有基准测试中的碰撞率均为零。另一方面，基于局部导航信息的技术在不同基准测试中存在一定数量的失败。此外，失败率随智能体数量或密度的增加而上升。

为更好地理解全局信息如何帮助导航系统，我们在窄走廊基准测试中使用 [39] 的方法计算了全局地图上的扰动显著性。结果如图 9 所示。全局地图中对决策最重要的区域包括：1）前方被其他智能体在局部激光扫描中遮挡的智能体；2）后方局部激光扫描无法覆盖的智能体；3）附近障碍物。因此，全局信息可帮助智能体提前规划，并对附近障碍物更加警惕。

### E. 可扩展性

不同数量机器人的运行时间如图 10 所示。可以观察到运行时间随智能体数量呈线性增长。这是因为我们方法的决策过程是独立的，每个智能体可以根据其接收到的信息自行计算动作。与大多数传统的基于几何的方法相比，分析全局环境的步骤可能具有超线性的时间复杂度，尤其对于本文使用的拥挤或具有挑战性的基准测试。例如，某些方法计算 K-近邻或使用 Voronoi 图计算环境的路线图，这可能具有超线性复杂度。相比之下，我们的方法不执行任何此类全局计算，仅利用神经网络的能力。此外，全局地图的计算时间复杂度为 $O(n)$。

> **图 10：** 不同数量智能体在若干基准测试中的运行时间。使用 32 核 CPU 和 NVIDIA RTX 2080 Ti 生成这些性能图。时间图表明本方法对数十个机器人而言是实用的。与仅使用局部信息的学习算法 [16] 相比，使用全局信息带来的额外运行时间开销很小。

---

## VI. 结论、局限性与未来工作

我们提出了一种基于深度强化学习的新颖多智能体导航算法。我们展示了环境的全局信息可以通过全局地图来利用，并提出了一种新颖的网络架构来为每个机器人计算无碰撞轨迹。我们在多个具有挑战性的场景中突出了其优势，并展示了相对于几何分散式方法或仅使用局部信息的强化学习方法的改进。此外，实验结果表明 DeepMNavigate 算法在密集和窄通道场景中相比先前方法能提供更好的性能。我们还在与训练场景不同的全新基准测试上展示了性能，证明了离散化地图中全局信息对基于 DRL 方法的价值。

研究结果令人鼓舞，有许多方式可以提升性能。我们的想法可以扩展到三维环境，通过将全局地图替换为三维版本并使用三维卷积神经网络进行处理。当前训练场景不包含动态和密集障碍物，未来可将其纳入。此外，需要更好的技术来计算全局地图和局部地图的最优大小，也可以包含速度、朝向或动力学约束等其他状态信息分量。还需要将方法扩展到没有关于移动障碍物运动信息的一般动态场景。本方法假设全局信息可用，而在许多场景中获取此类信息可能代价高昂。使用全局信息增加了训练计算的复杂度，我们使用两阶段算法来减少运行时间。另一种可能性是使用自编码器（auto-encoder）自动导出低维特征表示，然后将其用作特征提取器。

可以将全局和局部导航计算分离，以将本方法与局部方法结合来避免与其他智能体或动态障碍物的碰撞 [7]、[8]。这种局部和全局方法的结合已被用于模拟大型人群 [40]，开发类似的基于学习算法的框架可能是有益的。此外，使用局部或全局信息的基于 DRL 的方法也可以与全局导航数据结构（如路线图）结合，以进一步提升导航性能。

我们仅在具有挑战性的合成环境中展示了 DRL 方法的应用。未来一个重要的研究方向是扩展到真实世界场景，在那里需要使用其他技术来生成运动信息地图。将我们的学习方法与 SLAM 技术结合以提升导航能力将是有价值的。

---

## VII. 致谢

本工作获得 ARO 资助（W911NF1810313 和 W911NF1910315）及 Intel 支持。Tingxiang Fan 和 Jia Pan 部分获得香港特别行政区研究资助局优配研究金（GRF）HKU 11202119、11207818 的支持。

---

## 参考文献

[1] J. P. Van Den Berg and M. H. Overmars, "Prioritized motion planning for multiple robots," in IROS. IEEE, 2005, pp. 430–435.

[2] J. van Den Berg, J. Snoeyink, M. C. Lin, and D. Manocha, "Centralized path planning for multiple robots: Optimal decoupling into sequential plans." in Robotics: Science and systems, vol. 2, 2009.

[3] R. J. Luna and K. E. Bekris, "Push and swap: Fast cooperative path-finding with completeness guarantees," in IJCAI, 2011.

[4] G. Sanchez and J.-C. Latombe, "Using a prm planner to compare centralized and decoupled planning for multi-robot systems," in ICRA, vol. 2. IEEE, 2002, pp. 2112–2119.

[5] K. Solovey and D. Halperin, "On the hardness of unlabeled multi-robot motion planning," The International Journal of Robotics Research, vol. 35, no. 14, pp. 1750–1759, 2016.

[6] J. Yu and D. Rus, "An effective algorithmic framework for near optimal multi-robot path planning," in Robotics Research. Springer, 2018, pp. 495–511.

[7] J. Van Den Berg, S. J. Guy, M. Lin, and D. Manocha, "Reciprocal n-body collision avoidance," in Robotics research. Springer, 2011.

[8] R. Geraerts, A. Kamphuis, I. Karamouzas, and M. Overmars, "Using the corridor map method for path planning for a large number of characters," in International Workshop on Motion in Games, 2008.

[9] D. Helbing and P. Molnar, "Social force model for pedestrian dynamics," Physical review E, vol. 51, no. 5, p. 4282, 1995.

[10] D. Fox, W. Burgard, and S. Thrun, "The dynamic window approach to collision avoidance," IEEE Robotics & Automation Magazine, 1997.

[11] T. Fraichard and H. Asama, "Inevitable collision states — a step towards safer robots?" Advanced Robotics, 2004.

[12] T. Fan, P. Long, W. Liu, and J. Pan, "Fully distributed multi-robot collision avoidance via deep reinforcement learning for safe and efficient navigation in complex scenarios," arXiv, 2018.

[13] M. Pfeiffer, M. Schaeuble, J. Nieto, R. Siegwart, and C. Cadena, "From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots," arXiv e-prints, p. arXiv:1609.07910, Sep 2016.

[14] Y. F. Chen, M. Liu, M. Everett, and J. P. How, "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning," in ICRA. IEEE, 2017, pp. 285–292.

[15] M. Everett, Y. F. Chen, and J. P. How, "Motion planning among dynamic, decision-making agents with deep reinforcement learning," in IROS. IEEE, 2018, pp. 3052–3059.

[16] P. Long, T. Fan, X. Liao, W. Liu, H. Zhang, and J. Pan, "Towards optimally decentralized multi-robot collision avoidance via deep reinforcement learning," in ICRA. IEEE, 2018, pp. 6252–6259.

[17] P. Long, W. Liu, and J. Pan, "Deep-learned collision avoidance policy for distributed multiagent navigation," IEEE Robotics and Automation Letters, vol. 2, no. 2, pp. 656–663, 2017.

[18] R. Luna and K. E. Bekris, "Efficient and complete centralized multi-robot path planning," in IROS. IEEE, 2011, pp. 3268–3275.

[19] G. Sharon, R. Stern, A. Felner, and N. R. Sturtevant, "Conflict-based search for optimal multi-agent pathfinding," Artificial Intelligence, vol. 219, pp. 40–66, 2015.

[20] J. Yu and S. M. LaValle, "Optimal multirobot path planning on graphs: Complete algorithms and effective heuristics," IEEE Transactions on Robotics, vol. 32, no. 5, pp. 1163–1177, 2016.

[21] S. Tang, J. Thomas, and V. Kumar, "Hold or take optimal plan (hoop): A quadratic programming approach to multi-robot trajectory generation," IJRR, vol. 37, no. 9, pp. 1062–1084, 2018.

[22] J. Alonso-Mora, A. Breitenmoser, M. Rufli, P. Beardsley, and R. Siegwart, "Optimal reciprocal collision avoidance for multiple non-holonomic robots," in Distributed Autonomous Robotic Systems. Springer, 2013, pp. 203–216.

[23] U. Muller, J. Ben, E. Cosatto, B. Flepp, and Y. L. Cun, "Off-road obstacle avoidance through end-to-end learning," in Advances in neural information processing systems, 2006, pp. 739–746.

[24] J. Zhang, J. T. Springenberg, J. Boedecker, and W. Burgard, "Deep reinforcement learning with successor features for navigation across similar environments," in IROS. IEEE, 2017, pp. 2371–2378.

[25] A. Barreto, W. Dabney, R. Munos, J. J. Hunt, T. Schaul, H. P. van Hasselt, and D. Silver, "Successor features for transfer in reinforcement learning," in NIPS, 2017, pp. 4055–4065.

[26] J. Sergeant, N. Sunderhauf, M. Milford, and B. Upcroft, "Multimodal deep autoencoders for control of a mobile robot," in Proc. of Australasian Conf. for Robotics and Automation (ACRA), 2015.

[27] S. Ross, N. Melik-Barkhudarov, K. S. Shankar, A. Wendel, D. Dey, J. A. Bagnell, and M. Hebert, "Learning monocular reactive uav control in cluttered natural environments," in ICRA. IEEE, 2013.

[28] M. Pfeiffer, M. Schaeuble, J. Nieto, R. Siegwart, and C. Cadena, "From perception to decision: A data-driven approach to end-to-end motion planning for autonomous ground robots," in ICRA, 2017.

[29] L. Tai, G. Paolo, and M. Liu, "Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation," in IROS. IEEE, 2017, pp. 31–36.

[30] G. Kahn, A. Villaflor, V. Pong, P. Abbeel, and S. Levine, "Uncertainty-aware reinforcement learning for collision avoidance," arXiv, 2017.

[31] J. Godoy, I. Karamouzas, S. J. Guy, and M. L. Gini, "Moving in a crowd: Safe and efficient navigation among heterogeneous agents." in IJCAI, 2016, pp. 294–300.

[32] Y. F. Chen, M. Everett, M. Liu, and J. P. How, "Socially aware motion planning with deep reinforcement learning," in IROS, 2017.

[33] H.-J. Yoon, H. Chen, K. Long, H. Zhang, A. Gahlawat, D. Lee, and N. Hovakimyan, "Learning to communicate: A machine learning framework for heterogeneous multi-agent robotic systems," in AIAA Scitech 2019 Forum, 2019, p. 1456.

[34] A. Jagan Sathyamoorthy, J. Liang, U. Patel, T. Guan, R. Chandra, and D. Manocha, "Densecavoid: Real-time navigation in dense crowds using anticipatory behaviors," arXiv, pp. arXiv–2002, 2020.

[35] Wikipedia contributors, "Algorithms for calculating variance — Wikipedia, the free encyclopedia," 2020, [Online; accessed 31-May-2020]. [Online]. Available: https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=959752885

[36] L. Engstrom, A. Ilyas, S. Santurkar, D. Tsipras, F. Janoos, L. Rudolph, and A. Madry, "Implementation matters in deep rl: A case study on ppo and trpo," in ICLR, 2020.

[37] A. Nair, B. McGrew, M. Andrychowicz, W. Zaremba, and P. Abbeel, "Overcoming exploration in reinforcement learning with demonstrations," in ICRA. IEEE, 2018, pp. 6292–6299.

[38] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv, 2017.

[39] S. Greydanus, A. Koul, J. Dodge, and A. Fern, "Visualizing and understanding atari agents," arXiv preprint arXiv:1711.00138, 2017.

[40] R. Narain, A. Golas, S. Curtis, and M. C. Lin, "Aggregate dynamics for dense crowd simulation," in TOG. ACM, 2009.
