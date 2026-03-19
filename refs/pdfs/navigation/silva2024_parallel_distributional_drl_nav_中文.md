# 面向地面移动机器人无地图导航的并行分布式深度强化学习

**作者**: Victor A. Kich, Alisson H. Kolling, Junior Costa de Jesus, Gabriel V. Heisler, Hiago Jacobs, Jair A. Bottega, André L. da S. Kelbouscas, Akihisa Ohya, Ricardo B. Grando, Paulo L. J. Drews-Jr, Daniel F. T. Gamarra

**单位**:
1. 筑波大学智能机器人实验室
2. 南里奥格兰德联邦大学
3. 圣玛丽亚联邦大学
4. 乌拉圭科技大学

---

## 摘要

本文提出了基于并行分布式演员-评论家（actor-critic）网络的新型深度强化学习（Deep Reinforcement Learning, Deep-RL）技术，用于地面移动机器人的导航。所提方法利用激光测距数据、到目标的相对距离和角度来引导机器人。我们在 Gazebo 仿真器中训练智能体，并将其部署到真实场景中。结果表明，并行分布式 Deep-RL 算法提升了决策能力，在导航和空间泛化方面优于非分布式方法和基于行为的方法。

**关键词**: 并行分布式深度强化学习（Parallel Distributional Deep Reinforcement Learning）、地面移动机器人（Terrestrial Mobile Robot）、无地图导航（Mapless Navigation）

---

## 1. 引言

深度强化学习（Deep-RL）在工程和机器人领域的离散与连续系统控制中展现了巨大潜力 [1]。最初应用于稳定环境 [2]，但当涉及地面移动机器人等非平稳系统时，由于环境交互的复杂性，其难度显著增加。

为解决这一问题，研究者开发了专注于动作离散化的新型 Deep-RL 技术 [3]，在各类移动机器人的无地图导航中取得了成功 [4], [5], [6]。分布式 Deep-RL 方法为有限仿真环境中的训练时间问题提供了有前景的解决方案 [7]。然而，地面移动机器人在复杂环境中的自主导航仍然具有挑战性。

我们提出了两种基于并行分布式技术的新型 Deep-RL 方法：并行分布式确定性强化学习（Parallel Distributional Deterministic Reinforcement Learning, PDDRL）和并行分布式随机强化学习（Parallel Distributional Stochastic Reinforcement Learning, PDSRL），两者均结合了优先经验回放（Prioritized Experience Replay）以增强复杂场景中的导航能力。如图 1 所示，我们的方法使用 24 维测距数据以及到目标的相对距离和角度。多个智能体的同步学习提升了仿真和真实场景中的表现。我们使用 Turtlebot3 Burger 机器人在四个复杂度递增的场景中进行了广泛评估，并额外增加一个真实场景用于空间泛化测试。

本文的贡献包括：

- 提出两种分布式 Deep-RL 方法，利用简单的基于测距的感知架构改善面向目标的无地图导航。
- 通过仿真到真实（sim-to-real）的评估验证了所提方法的可行性，解决了不精确性和延迟等挑战。
- 证明了带有优先经验回放的随机演员-评论家技术优于非分布式技术和经典算法，这是首次针对地面移动机器人无地图导航使用并行分布式 Deep-RL 方法进行广泛的仿真到真实评估。

> **图 1**: 左图：Turtlebot3 Burger 在真实障碍物场景中导航。右图：所提 PDDRL 和 PDSRL 方法的输入输出结构。输入包括激光测距数据、到目标的距离和角度；输出为线速度和角速度。确定性技术对应 PDDRL，随机技术对应 PDSRL。

---

## 2. 相关工作

传统 Deep-RL 将奖励建模为单一值，但 Bellemare 等人 [8] 提出将其建模为概率分布。这一思想被扩展到 Soft Actor-Critic（SAC）算法的新实现中，由 Duan 等人 [9] 提出，启发了我们的方法。

分布式 Deep-RL 的引入旨在加速训练，通过将计算分布到多个处理器上实现 [10]。Mnih 等人 [11] 使用异步演员-评论家方法，而 Horgan 等人开发了 Ape-X 方法，将演员与学习者解耦 [10]。Barth-Maron 等人 [7] 通过深度确定性策略梯度（Deep Deterministic Policy Gradient, DDPG）算法进一步改进了这一方法。

Tai 等人 [12] 展示了 Deep-RL 在移动机器人中的应用，推动了地面机器人无地图导航的进展 [3]。Jesus 等人 [4] 在无并行网络的情况下探索了无地图导航中的确定性和随机方法。

本文引入了面向地面移动机器人无地图导航的并行 Deep-RL 方法，在更复杂的场景中进行仿真到真实的评估。我们提出了 PDDRL（确定性）和 PDSRL（随机）方法，并与传统的基于行为的算法（Behavior-Based Algorithm, BBA）[13] 以及使用经典和优先记忆回放的 DDPG 和 SAC 的并行版本进行了比较。

---

## 3. 方法

本节介绍 Deep-RL 的核心概念，涵盖确定性和随机方法的细节。此外，我们展示了仿真和真实场景，并讨论了奖励系统和网络架构等具体问题。

### 3.1 深度强化学习

标准强化学习的目标是最大化折扣奖励总和的期望。状态-动作值函数是一个数学模型，描述从状态 $s$ 采取动作 $a$ 后按照策略 $\pi$ 行动所获得的期望回报 [14]。该函数定义为：

$$Q^{\pi}(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right], \tag{1}$$

通常用于评估策略的质量。

在每个时间步，网络预测当前状态的行为并生成时间差分（Temporal Difference, TD）误差信号。贝尔曼算子（Bellman operator）

$$(T^{\pi}Q)(s, a) = r(s, a) + \gamma \mathbb{E}[Q(s', \pi(s')) \mid s, a] \tag{2}$$

可以最小化该 TD 误差，其中期望关于下一状态 $s'$ 计算。

本工作使用两个独立的神经网络——演员网络和评论家网络——来评估 TD 误差。TD 误差在具有独立参数 $(\theta', \omega')$ 的目标策略和值网络下进行评估，以稳定学习过程。评论家网络生成动作的 Q 值，而演员网络的输出是表示所选动作的实数值。

### 3.2 并行分布式确定性强化学习

由 [15] 提出并在 D4PG [7] 中扩展的 DDPG 架构是 Deep-RL 在连续观测空间中应用于移动机器人的基石 [16]。它采用演员-评论家框架，利用近似函数在连续空间中学习策略。

DDPG 与 D4PG 的主要区别在于后者引入了分布式贝尔曼算子（Distributional Bellman Operator），这对优化过程至关重要。该算子定义如下：

$$(T^{\pi}Z)(s, a) = r(s, a) + \gamma \mathbb{E}[Z(s', \pi(s')) \mid s, a], \tag{3}$$

其中 $Q^{\pi}(s, a) = \mathbb{E}[Z^{\pi}(s, a)]$，$Z^{\pi}$ 返回一个分布式随机变量，即分类分布（categorical distribution）。

在 D4PG 框架中，分类分布对评论家网络的输出进行建模，预测预定义奖励区间上的概率向量，每个区间对应一个潜在奖励值范围。该分布使网络能够估计期望回报的完整概率分布，而非单一期望值，从而捕获可能结果的变异性：

$$\text{Categorical}(Z^{\pi}(s, a)) = [p_1, p_2, \ldots, p_k], \tag{4}$$

其中 $p_i$ 是回报落入第 $i$ 个区间的概率，$k$ 为区间数量。评论家的训练涉及最小化预测分布与目标分布之间的散度，通过更好地表征动态环境中的不确定性来增强策略的鲁棒性。

此外，D4PG 引入了 N 步回报（N-step returns）的概念来估计 TD 误差，表示为：

$$(T^N_{\pi}Q)(s, a) = r(s, a) + \mathbb{E}\left[\sum_{n=1}^{N-1} \gamma^n r(s_n, a_n) + \gamma^N Q(s_N, \pi(s_N)) \mid s, a\right]. \tag{5}$$

在本研究中，我们基于 D4PG 开发了名为 PDDRL 的方法，融合了 N 步回报的扩展以及建模为分类分布的评论家值函数 [8]。确定性策略记为 $\mu$，带噪声的动作表示为 $\mu'$，其噪声过程为：

$$\mu' = \mu(s_t) + \mathcal{N}, \tag{6}$$

其中 $\mathcal{N}$ 服从 Ornstein-Uhlenbeck 过程 [17]。

鉴于原始 D4PG 研究 [7] 中报告的实验结果不一致，我们开发了两个变体：使用经典回放记忆的 PDDRL 和使用优先回放记忆 [18] 的 PDDRL-P。

最后，需要建立目标网络以增强学习稳定性。该网络是演员和评论家网络的副本，但使用"软"更新。目标网络的权重 $\theta'$ 根据因子 $\tau$ 逐步调整：

$$\theta' = \tau\theta + (1 - \tau)\theta'. \tag{7}$$

### 3.3 并行分布式随机强化学习

SAC 架构 [19] 作为 DDPG 等确定性演员-评论家方法的随机对应版本，利用近似函数在连续动作空间中学习策略。SAC 引入了带有熵增强的贝尔曼算子，有助于探索和策略优化：

$$(T^{\pi}Q)(s, a) = r(s, a) + \gamma \mathbb{E}\left[Q(s', a') - \alpha \log \pi(a'|s') \mid s, a\right]. \tag{8}$$

为促进鲁棒探索，SAC 强调同时最大化奖励和策略熵，从而促进动作多样性并防止策略过早收敛。该算法对产生相似 Q 值的动作赋予相同概率，并缓解由于不确定动作导致的 Q 函数近似失败风险。

在此基础上，DSAC [20] 将 SAC 的最大熵框架与 DDPG 的分布式方法融合，产生混合贝尔曼算子：

$$(T^{\pi}Z)(s, a) = r(s, a) + \gamma\left[Z(s', a') - \alpha \log \pi(a'|s') \mid s, a\right], \tag{9}$$

该算子整合了随机和分布式组件，进一步描述为：

$$Z^{\pi}(s, a) = \sum_{t=0}^{\infty} [r(s, a) + \gamma - \alpha \log \pi(a_{t+1}|s_{t+1}) \mid s, a]. \tag{10}$$

在本研究中，新提出的 PDSRL 方法建立在 DSAC 之上，结合了 N 步回报和软更新——这些特性在第 3.2 节 PDDRL 部分已详细介绍——以提高预测精度和稳定性。

我们实现了两个变体：使用经典回放记忆的 PDSRL 和使用优先回放记忆的 PDSRL-P，两者均采用 100000 步大小的回放记忆，以确保所有 Deep-RL 方法的一致性。

### 3.4 网络结构

本文所有方法均采用具有 26 个输入和 2 个输出的神经网络。输入包括激光雷达（Lidar）的 24 个测距值，以及到目标的相对位置和相对角度。传感器采样范围从 0° 到 360°，等间距 15°。输入角度用于引导车辆朝向目标并增强学习过程，而目标距离鼓励网络将其最小化。输出为线速度和角速度，用于控制车辆。作为与 Deep-RL 方法的对比，我们采用了传统的基于行为的算法（BBA），该算法使用硬编码的导航程序。需要注意的是，为公平比较，BBA 使用与其他方法相同数量的传感器距离信息。

所有方法使用的神经网络具有三个隐藏全连接层，每层 256 个神经元，通过 ReLU 激活函数连接。动作范围在 -1 到 1 之间，演员网络的激活函数为双曲正切函数（Tanh）。线速度输出缩放在 -0.12 到 0.15 米之间，角速度输出缩放在 -0.1 m/s 到 0.1 m/s 之间。在两种方法中，评论家网络预测当前状态的 Q 值，演员网络预测当前状态。在并行方法的训练阶段，使用四个智能体进行训练，一个智能体用于评估。训练过程如图 2 所示。

> **图 2**: 并行 Deep-RL 训练过程结构。包含探索工作器（Exploration Worker）、采样工作器（Sampling Worker）、学习器工作器（Learner Worker）和评估工作器（Evaluation Worker）。探索工作器产生转移（transition）存入回放缓冲区；采样工作器从缓冲区采样批次送入学习器；学习器通过优化器更新权重并将新策略权重分发回探索工作器；评估工作器用于评估当前策略。

### 3.5 奖励函数

为促进学习，必须建立一个奖励系统，鼓励智能体的良好行为并惩罚不良行为。该系统基于经验评估获得的经验知识设计。本文实现的奖励系统如下：

$$r(s_t, a_t) = \begin{cases} r_{\text{arr}} & \text{if } d_t < c_d \\ r_{\text{coll}} & \text{if } \min_x < c_o \\ r_{\text{idle}} & \text{if } \min_x \geq c_o \text{ and } d_t \geq c_d \end{cases} \tag{11}$$

定义了一个简单的奖励函数，仅提供三种类型的奖励：成功完成任务的奖励、未能完成任务的惩罚，以及两者均未发生时的中间奖励。当智能体在 $c_d$ 米范围内成功到达目标时给予 200 的奖励，其中 $c_d$ 设为 0.25 米。如果智能体与障碍物碰撞或到达场景边界，给予 -20 的负奖励 $r_{\text{coll}}$。当距离传感器读数小于 $c_o$（设为 0.12 米）时检测为碰撞。如果智能体的激光雷达距离大于 $c_o$ 且目标距离大于 $c_d$，则给予 0 的奖励 $r_{\text{idle}}$。该函数使研究能够聚焦于 Deep-RL 方法本身的异同，而非场景差异。

---

## 4. 实验

本节展示了所提方法的实验评估。实验旨在评估方法在仿真和真实环境中的性能。我们描述了用于训练和测试的仿真与真实场景的设置，随后对实验结果进行了详细分析。

### 4.1 仿真与真实场景

机器人和场景的仿真使用 Gazebo 仿真器。机器人与智能体之间的连接通过机器人操作系统（Robot Operating System, ROS）实现。为分配智能体，并行创建多个 Gazebo 仿真器实例，采集的数据发布到回放缓冲区。所选机器人为 TurtleBot3 Burger 版本。真实和仿真的 Turtlebot3 Burger 如图 3 所示。

> **图 3**: 仿真和真实实验设置。展示了四个递增复杂度的场景（标记为 1-4），上排为仿真场景，下排为对应的真实场景。

本研究使用了四个仿真场景。第一个场景为机器人提供可导航区域，墙壁是唯一可能导致碰撞的障碍物。碰撞墙壁或任何障碍物时，对该动作给予负奖励并终止当前回合。第二个场景包含四个固定的圆柱形障碍物，每个半径 25 厘米。第三和第四个场景旨在为机器人到达最终目标创建更具挑战性的路径。第三个场景包含一个"U"形物体，为智能体提供两条等代价的可能轨迹，障碍物在中间形成死胡同。第四个场景不对称且更复杂，要求智能体开发更好的策略以避免碰撞。真实和仿真场景均在图 3 中标注。

在通过仿真训练方法后，我们在真实场景中评估了所提方法。真实场景中的部分必要数据通过 OpenCV 图像处理获得。四个真实场景与仿真场景相似，但几何形状并不完全相同。

### 4.2 实验结果

我们在仿真和真实场景中评估了所有方法的导航能力和空间泛化能力。训练阶段包括第一个场景的 30000 步和其他场景的 200000 步。训练完成后，我们分析了奖励的移动平均值（图 4）。结果表明，随机分布式方法在所有场景中优于其他方法，尽管起步较慢。DDPG 的奖励均值与表现最佳的方法相近。每种方法在预定目标坐标下进行了 100 回合的评估。

> **图 4**: 所有并行方法在每个场景中各训练步的智能体奖励移动平均值。场景按从左到右、从上到下的顺序排列：场景 1、2、3、4。包含 DDPG、DDPG-P、PDDRL、PDDRL-P、PDSRL、PDSRL-P、SAC、SAC-P 八种方法的训练曲线。

> **图 5**: 通过在每个仿真场景中进行 100 次导航试验评估各并行方法的行为。线条展示了智能体的行驶路径，每个智能体有 25 次尝试捕获每个目标。展示了 DDPG、SAC、DDPG-P、SAC-P、PDDRL、PDSRL、PDDRL-P、PDSRL-P 在四个场景中的轨迹。

> **图 6**: BBA 方法在仿真和真实场景中的评估，分别进行 100 次和 12 次试验。线条表示智能体的路径，仿真中每个目标 25 次尝试，真实场景中每个目标 3 次尝试。

**表 1**: 所有方法在四个不同场景中 100 次（仿真）和 12 次（真实）导航试验的精度。

| 场景 | BBA | DDPG | SAC | DDPG-P | SAC-P | PDDRL | PDSRL | PDDRL-P | PDSRL-P |
|------|-----|------|-----|--------|-------|-------|-------|---------|---------|
| 仿真场景 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 仿真场景 2 | 62% | 100% | 0.0% | 85% | 64% | 100% | 100% | 100% | 100% |
| 仿真场景 3 | 75% | 74% | 0.0% | 75% | 0.0% | 100% | 100% | 70% | 100% |
| 仿真场景 4 | 73% | 81% | 25% | 92% | 0.0% | 83% | 98% | 76% | 100% |
| 真实场景 1 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 真实场景 2 | 83.3% | 100% | 0.0% | 100% | 41.6% | 100% | 91.6% | 100% | 100% |
| 真实场景 3 | 75% | 100% | 0.0% | 100% | 0.0% | 100% | 100% | 0.0% | 100% |
| 真实场景 4 | 75% | 66.6% | 25% | 100% | 25% | 25% | 75% | 75% | 100% |

表 1 显示了仿真实验结果。图 5 和图 6 展示了 Deep-RL 和经典方法的行为。在场景 1 中，所有方法均达到 100% 精度。在更复杂的场景中，分布式算法保持了 100% 精度，而其他方法表现不一。DDPG 的表现与分布式算法相当。

在场景 3 中，分布式方法优于其他方法，但 PDDRL-P 仅有 70% 精度。这一问题在文献中已知，优先记忆可能限制基于 D4PG 算法的确定性分布式智能体的泛化能力。在场景 4 中，仅 PDSRL-P 方法捕获了所有点，PDSRL 以 98% 精度紧随其后。值得注意的是，DDPG-P 达到了 92% 精度，超过了确定性分布式算法。

图 5 显示 PDSRL 和 PDSRL-P 具有更平滑的轨迹。DDPG 在场景 3 中表现出不稳定性，反复碰撞。优先化在场景 4 中减少了碰撞，但不稳定性依然存在。真实环境实验使用了与仿真类似的目标点。

> **图 7**: （上）通过在每个真实场景中进行 12 次导航试验评估各并行方法的行为。线条展示了智能体的行驶路径，每个智能体有 3 次尝试捕获每个目标。（下）在额外真实场景中评估排名靠前的并行方法的行为。按以下顺序排列：DDPG、DDPG-P、PDDRL、PDSRL、PDSRL-P。

PDSRL-P 和 DDPG-P 在真实环境中达到了 100% 精度。PDDRL 和 PDDRL-P 在场景 3 和 4 中的性能差异显著。真实场景引入了延迟，影响了性能。

**表 2**: 表现最佳方法在额外场景中 12 次导航试验的精度和距离指标。

| 算法 | 精度 (%) | 碰撞率 (%) |
|------|---------|-----------|
| DDPG | 58.3% | 41.6% |
| DDPG-P | 100% | 0.0% |
| PDDRL | 100% | 0.0% |
| PDSRL | 75% | 8.3% |
| PDSRL-P | 100% | 0.0% |

我们提出了一个新颖的场景，将场景 3 中预训练的模型在未见过的环境中进行评估。图 7 和表 2 展示了结果。DDPG 具有较高的碰撞率，而 PDSRL 最小化了碰撞但并非总能获得奖励。DDPG-P、PDDRL 和 PDSRL-P 达到了 100% 精度，展现了强大的空间泛化能力。

我们的广泛验证表明，所提方法在所有场景中均表现良好，包括新颖场景。Deep-RL 方法在真实世界挑战中优于传统算法。优先化版本取得了最佳成功率，尤其是 PDSRL-P。在新颖场景中的评估确认了所提方法的学习能力和空间泛化能力。

---

## 5. 结论

本文引入了面向地面移动机器人无地图导航的新型深度强化学习（Deep-RL）技术，展示了其在仿真中的学习能力以及在真实世界中实施的潜力。结果表明，并行分布式 Deep-RL 方法能够有效解决复杂的真实世界机器人问题，而非学习型方法和不带优先记忆回放的非分布式方法无法应对这些问题。

---

## 参考文献

[1] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, "Continuous control with deep reinforcement learning," in ICLR. PMLR, 2016.

[2] S. Gu, T. Lillicrap, I. Sutskever, and S. Levine, "Continuous deep q-learning with model-based acceleration," in ICML, 2016, pp. 2829–2838.

[3] Y. Zhu, R. Mottaghi, E. Kolve, J. J. Lim, A. Gupta, L. Fei-Fei, and A. Farhadi, "Target-driven visual navigation in indoor scenes using deep reinforcement learning," in IEEE ICRA. IEEE, 2017, pp. 3357–3364.

[4] J. C. de Jesus, V. A. Kich, A. H. Kolling, R. B. Grando, M. A. d. S. L. Cuadros, and D. F. T. Gamarra, "Soft actor-critic for navigation of mobile robots," Journal of Intelligent & Robotic Systems, vol. 102, no. 2, pp. 1–11, 2021.

[5] R. B. Grando, J. C. de Jesus, V. A. Kich, A. H. Kolling, N. P. Bortoluzzi, P. M. Pinheiro, A. A. Neto, and P. L. Drews, "Deep reinforcement learning for mapless navigation of a hybrid aerial underwater vehicle with medium transition," in IEEE ICRA. IEEE, 2021, pp. 1088–1094.

[6] R. B. Grando, J. C. de Jesus, V. A. Kich, A. H. Kolling, and P. L. J. Drews-Jr, "Double critic deep reinforcement learning for mapless 3d navigation of unmanned aerial vehicles," Journal of Intelligent & Robotic Systems, vol. 104, no. 2, pp. 1–14, 2022.

[7] G. Barth-Maron, M. W. Hoffman, D. Budden, W. Dabney, D. Horgan, D. Tb, A. Muldal, N. Heess, and T. Lillicrap, "Distributed distributional deterministic policy gradients," ICLR, 2018.

[8] M. G. Bellemare, W. Dabney, and R. Munos, "A distributional perspective on reinforcement learning," in ICML. PMLR, 2017, pp. 449–458.

[9] J. Duan, Y. Guan, S. E. Li, Y. Ren, Q. Sun, and B. Cheng, "Distributional soft actor-critic: Off-policy reinforcement learning for addressing value estimation errors," IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2021.

[10] D. Horgan, J. Quan, D. Budden, G. Barth-Maron, M. Hessel, H. van Hasselt, and D. Silver, "Distributed prioritized experience replay," in ICLR. PMLR, 2018.

[11] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, "Asynchronous methods for deep reinforcement learning," in ICML, vol. 48. New York, New York, USA: PMLR, 20–22 Jun 2016, pp. 1928–1937.

[12] L. Tai, G. Paolo, and M. Liu, "Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation," in IEEE/RSJ IROS. IEEE, 2017, pp. 31–36.

[13] V. J. Lumelsky and A. A. Stepanov, "Path-planning strategies for a point mobile automaton moving amidst unknown obstacles of arbitrary shape," Algorithmica, vol. 2, no. 1, pp. 403–430, Nov 1987.

[14] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press, 2018.

[15] D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller, "Deterministic policy gradient algorithms," in ICML. PMLR, 2014, pp. 387–395.

[16] L. Tai and M. Liu, "Towards cognitive exploration through deep reinforcement learning for mobile robots," 2016.

[17] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of the brownian motion," Physical review, vol. 36, no. 5, p. 823, 1930.

[18] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, "Prioritized experience replay," in ICLR. PMLR, 2016.

[19] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel et al., "Soft actor-critic algorithms and applications," 2018.

[20] X. Ma, L. Xia, Z. Zhou, J. Yang, and Q. Zhao, "Dsac: Distributional soft actor critic for risk-sensitive reinforcement learning," 2020.
