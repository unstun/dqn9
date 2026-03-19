# 基于深度强化学习的自主移动机器人在综合未知环境中的路径规划

**IEEE INTERNET OF THINGS JOURNAL, VOL. 11, NO. 12, 2024年6月15日**

**作者：** Zekun Bai, Hui Pang (IEEE 会员), Zhaonian He, Bin Zhao, Tong Wang

**通讯作者：** Hui Pang (panghui@xaut.edu.cn)

**单位：** 西安理工大学机械与精密仪器工程学院，西安 710048

**DOI:** 10.1109/JIOT.2024.3379361

---

## 摘要

在实际应用场景中，自主移动机器人（Autonomous Mobile Robot, AMR）在未知环境下的路径规划往往面临一些不可避免的问题，如对环境信息的高度依赖、推理时间过长以及抗干扰能力不足。为解决上述问题，本文提出一种改进的基于深度强化学习（Deep Reinforcement Learning, DRL）的路径规划算法，为一类AMR寻找最优路径。首先，将AMR的路径规划描述为马尔可夫决策过程（Markov Decision Process, MDP）框架，并利用双深度Q网络（Double Deep Q Network, DDQN）获取AMR路径规划的最优自适应解。其次，设计了一种融合启发式函数（Heuristic Function）的综合奖励函数，以引导AMR到达目标区域。随后，设计了一种带有自适应 $\varepsilon$-贪心（$\varepsilon$-greedy）动作选择策略的优化深度神经网络（Deep Neural Network, DNN），以处理探索与利用之间的权衡问题，从而进一步提高AMR路径规划的全局搜索能力和收敛性能。此外，利用贝塞尔曲线（Bezier Curve）理论对规划路径进行平滑处理。最后，通过对比仿真实验验证了所提路径规划算法的有效性。结果表明，与DQN、A\*、RRT和APF算法相比，改进的DDQN算法能够在综合未知环境中生成更安全、更短的全局路径。同时，IDDQN算法对未知环境中的随机干扰具有较强的适应性。

**关键词：** 自主移动机器人（AMR）、深度强化学习（DRL）、双深度Q网络（DDQN）、路径规划、路径平滑

---

## I. 引言

近年来，自主移动机器人（AMR）凭借其体积小、部署灵活、能够独立完成任务等优点，在生物医学、空间探测、工程设备检修维护等领域获得了广泛的应用前景 [1]。路径规划是AMR领域的核心功能之一，因此提高AMR的自主路径规划能力对于研究人员而言尤为重要。路径规划系统需要在满足多种约束条件的同时生成一条可行路径，这些约束包括：与障碍物保持安全距离、到达目标的最短距离、路径平滑性以及环境干扰信息 [2], [3]。当AMR在未知环境中执行任务时，应根据从周围环境获取的信息计算并修正其无碰撞的可行路径 [4]。此外，AMR还需要在外部环境因素可能产生干扰的情况下生成符合约束条件的路径导航信息。因此，具有高度自主性和学习能力的路径规划方法对AMR至关重要。

大量关于机器人系统路径规划算法的研究已被报道，这些算法大致可分为三类：1）传统算法；2）仿生算法；3）基于学习的算法，如图1所示。传统算法包括基于图的方法、基于采样的方法和基于数据的方法。仿生算法包括遗传算法（Genetic Algorithm）、人工蜂群算法（Artificial Colony Algorithm）、模糊逻辑算法（Fuzzy-Logic Algorithm）等。基于学习的算法包括深度强化学习（DRL）算法和元学习（Meta Learning）等。

> **图1.** 路径规划算法分类图。（展示了传统算法、仿生算法和基于学习的算法三大类别的层次结构。）

具体而言，常用的传统方法包括A\*算法 [5]、人工势场算法（Artificial Potential Field, APF）[6] 和快速探索随机树算法（Rapidly-exploring Random Trees, RRT）[7]。虽然A\*算法能够高效地确定最短路径，但其计算量会随着空间的扩展呈指数增长，因此不适用于复杂的连续障碍物环境 [8]。APF算法利用势函数下降方法来寻找理想路径 [9]。Lin等 [10] 设计了一种两层路径规划方法，采用改进的APF算法和动态窗口算法来引导机器人避障并合理规划更短、更平滑的路径。然而，随着障碍物数量的增加，APF算法在环境中将遇到越来越多的零势能点，容易陷入局部极小值而无法规划出完整路径。RRT算法是一种基于随机采样的路径规划算法，虽然具有较强的搜索能力和较快的搜索速度，但存在搜索精度低、路径平滑性差等局限 [11]。仿生算法利用生物结构的行为来寻找最优路径 [12]，典型算法包括遗传算法 [13]、蚁群优化（Ant Colony Optimization）[14] 等。与传统算法相比，仿生算法计算量更小，在处理复杂环境时具有明显优势。但它们仍存在一些不足，如对环境的强依赖性以及容易陷入局部最优。

因此，传统方法和仿生算法在缺乏全局先验环境信息的情况下难以解决路径规划问题，且对系统扰动等不确定性缺乏适应能力。为了在未知环境中实现路径规划，一些研究者开始开发更智能的路径规划方法，使AMR能够在实践中自主学习。机器学习（Machine Learning, ML）是发展最快的人工智能算法之一，旨在通过从历史经验数据中学习来自动改进系统策略 [15]。Liu等 [16] 设计了一种微型ML算法，协助无人机互联网在飞行过程中进行实时决策，最终生成高效安全的飞行路径。因此，将ML方法应用于机器人导航领域可以使机器人学习环境信息并自主完成路径规划任务。

强化学习（Reinforcement Learning, RL）作为ML的一个分支，能够通过与外部环境的交互并从外部环境获取反馈信息来学习和调整学习策略，在克服对环境信息的依赖方面展现出巨大潜力 [17], [18]。文献 [19] 使用Q-Learning算法处理未知环境中自主水下航行器的避障和路径规划问题。鉴于RL算法在未知环境中具有更好的路径规划能力，一些研究已成功将RL应用于移动机器人系统。例如，文献 [20] 提出了一种将Q-Learning与FPA相结合的算法，并将其应用于AMR的路径规划，解决了传统Q-Learning算法收敛速度慢的问题。文献 [21] 在基于Dyna架构的Q-Learning中引入了启发式搜索策略和模拟退火机制，增强了移动机器人路径规划的全局搜索性能和学习效率。然而，Q-Learning算法通过建立Q值表来存储环境中所有的状态值，并通过查询来访问这些信息。随着环境和动作维度的增加，Q-Learning算法需要大量的计算资源，降低了学习效率，并可能阻碍算法收敛 [22]。

深度强化学习（DRL）融合了RL的决策能力和深度学习（Deep Learning, DL）的感知能力，能够弥合高维输入与动作之间的鸿沟。近年来，DRL已广泛应用于无人机、自主水下航行器和无人水面艇的路径规划控制 [23]–[26]，以及自动驾驶 [27]。此外，学术界还提出了多种基于DRL的AMR路径规划算法，这些算法本质上基于深度Q网络（Deep Q Network, DQN）及其相关算法。具体而言，Wen等 [28] 提出了一种基于神经网络RL路径系统的新型路径规划方法，使移动机器人能够在不与任何障碍物或机器人发生碰撞的情况下导航到终端区域，该方法已成功应用于移动机器人实验平台。Wu等 [29] 提出了一种用于移动机器人自主转向的改进DDQN算法，并在各类真实世界障碍物环境中验证了其性能。Wang等 [30] 提出了一种基于HFG-DRL的新型自由步态多接触路径规划方法，训练后的路径规划策略能使机器人在非结构化环境中自动调整步态模式，快速平稳地到达目标区域。Tao和Hafid [31] 提出了基于DDQN-PER算法的DeepSensing框架，以解决深度感知任务分配问题，从而为移动用户规划更高效的行进路径并实现特定目标。Jiang等 [32] 提出了一种改进的DQN算法，以解决小行星表面复杂地形下跳跃式巡视器的路径规划问题，并证明了该路径规划算法在随机变化地形下具有一定的适应性。

然而，将DQN算法直接应用于路径规划会产生若干固有问题：1）随着环境模型复杂度的增加，DQN的学习效率将变差、收敛速度将下降；2）传统DQN算法规划的路径在安全性方面往往表现不佳，AMR存在与障碍物边缘碰撞的风险；3）传统DQN算法规划的路径轨迹往往存在冗余且缺乏平滑性。

受上述讨论的启发，本研究提出一种基于改进双深度Q网络（Improved DDQN, IDDQN）的算法，用于AMR在带有随机障碍物的综合未知环境中执行路径规划任务。本文的主要贡献总结如下：

1. 设计了一种新型综合奖励函数，使AMR能够在综合未知环境中快速且安全地接近目标区域。
2. 设计了一种自适应 $\varepsilon$-贪心动作策略，以解决探索与利用之间的权衡问题。该策略结合优化的深度神经网络（DNN），可以提高AMR路径规划的学习效率和收敛速度。
3. 多组仿真实验结果表明，所提出的IDDQN算法在解决AMR路径规划问题时表现优于传统方法。IDDQN算法在具有随机干扰的未知环境中的鲁棒性也得到了验证。

本文其余部分安排如下：第II节描述AMR的路径规划问题并介绍MDP和DDQN算法；第III节提出AMR路径规划的MDP模型以及基于IDDQN的AMR路径规划算法模型；第IV节给出不同路径规划算法的仿真对比实验分析，并验证IDDQN算法在扰动环境中的鲁棒性；第V节给出结论和未来工作展望。

---

## II. AMR路径规划基本理论

AMR路径规划如图2所示，是指AMR在考虑所有外部环境信息和多种约束条件的前提下，从起始点（Start Point, SP）到目标点（Target Point, TP）自动生成一条平滑且无碰撞的最优路径（绿色轨迹线）的完整过程 [33]。

> **图2.** AMR路径规划示意图。（展示了AMR从起始点到目标点在障碍物环境中规划路径的概念图。）

根据相关文献 [34], [35]，AMR的路径规划可以被视为MDP框架中的学习过程。其中，AMR采取动作与外部环境交互并改变其状态以获得奖励；同时，AMR的目标是学习获得最大累积奖励的动作策略，最终生成所需的路径轨迹。基于上述描述，首先需要介绍MDP理论和DDQN算法的相关知识。

### A. 马尔可夫决策过程

MDP可以表示为一个五元组 $M = [S, A, P, R]$，其中 $S$ 表示环境中存在的有限状态集，$s_t$ 表示时刻 $t$ 的状态，$A$ 表示智能体（AMR）执行的动作，$a_t$ 表示时刻 $t$ 执行的动作，$P$ 是转移概率，$R$ 是奖励函数。基于观察到的环境状态 $s_t$，AMR随机选择并执行动作 $a_t$，随后环境为AMR提供奖励 $r_{t+1}$，同时将状态从 $s_t$ 更新为下一状态 $s_{t+1}$。

> **图3.** MDP框架中AMR与环境的交互示意图。（展示了智能体在MDP框架中与环境进行状态-动作-奖励循环交互的过程。）

具体而言，在时刻 $t$，当AMR在状态 $s \in S$ 中执行动作 $a \in A$ 时，转移概率 $s'$ 可由下式给出：

$$P(s, a, s') = P(s'|s, a) = \text{prob}(s_{t+1} = s' | S_t = s, A_t = a) \tag{1}$$

在整个学习过程中，AMR的目标是学习与最大化长期累积奖励相对应的最优策略。AMR在探索过程中可以获得最大长期累积奖励 $R_t$，其定义为：

$$R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \tag{2}$$

其中 $r_t$ 表示AMR在时刻 $t$ 的奖励，$\gamma$ 是折扣因子（$\gamma \in [0,1]$），$\gamma$ 决定了未来回报的价值。

为了获得当前动作与未来奖励之间的长期关系，利用最优策略 $\pi$ 下的总潜在奖励来构建动作价值函数（Action-Value Function）$Q^\pi(s, a)$，其表达式为：

$$Q^\pi(s, a) = E_\pi[R_t | S_t = s, A_t = a] = E_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | s_t, a_t\right] \tag{3}$$

其中 $E_\pi$ 表示在概率分布 $\pi$ 下随机变量的数学期望。

### B. 双深度Q网络

本节利用DQN算法求解AMR路径规划决策问题。DQN作为DRL中的一种重要算法，有效地将Q-Learning与神经网络技术相结合，能够从高维复杂状态空间中获取最优控制策略，而无需依赖环境的先验信息 [36]。因此，利用DQN能使AMR在具有不完整外部环境信息的复杂障碍物环境中生成最优路径。

DQN算法使用神经网络来近似如式(3)所示的Q值函数。DQN算法的整体学习架构如图4所示。值得注意的是，目标网络（Target Network）和经验回放（Experience Replay）机制的引入显著增强了训练的准确性和稳定性 [37]。

> **图4.** DQN整体架构。（展示了包含当前网络、目标网络和经验回放池的DQN学习架构。）

在训练过程中，当前网络通过与环境的交互学习生成经验数据（当前状态 $s$、动作 $a$、奖励 $r$ 和下一状态 $s'$），并将其存储在具有一定容量的经验回放池中（若存储已满，则从第一组开始按顺序覆盖原始经验）。根据当前网络获得的经验数据，目标网络的参数不断更新和优化，最终获得具有最大奖励的动作策略。

当前网络和目标网络的区别在于输入向量的选择方式和网络更新周期。具体而言，当前网络以当前状态 $s$ 作为输入，在每次迭代时更新；目标网络以下一状态 $s'$ 作为输入，并定期将当前网络的参数复制到目标网络。在每个训练步中，使用时序差分（Temporal Difference）方法计算目标网络对应的目标Q值（$Q^{DQN}_{Target}$），其表达式为：

$$Q^{DQN}_{Target} = r + \gamma \max_{a'} Q(s', a'; \theta^-) \tag{4}$$

其中 $\max_{a'} Q(s', a'; \theta^-)$ 是目标网络输出的最大Q值，$a'$ 是AMR在状态 $s'$ 中采取的动作，$\theta^-$ 表示目标网络的权重参数。

当前网络预测的Q值与目标网络预测的Q值之间的均方差被定义为损失函数 $L(\theta)$，其形式为：

$$L(\theta) = E\left[\left(Q^{DQN}_{Target} - Q(s, a; \theta)\right)^2\right] \tag{5}$$

其中 $Q(s, a; \theta)$ 表示当前网络的输出，$\theta$ 表示当前网络的权重参数。

通过将DQN算法中目标Q值（$Q^{DQN}_{Target}$）的计算与动作选择解耦，DDQN算法可以进一步消除可能的过估计（Overestimation）问题，生成更可靠、更稳定的学习过程 [38]。

DDQN不是直接从目标网络获取最大Q值，而是首先根据当前网络找到最大Q值对应的动作，然后使用参数为 $\theta^-$ 的目标网络计算该动作对应的目标Q值（$Q^{DDQN}_{Target}$），如式(6)所示，所选动作的评估为：

$$Q^{DDQN}_{Target} = r + \gamma Q'(s', \arg\max_{a'} Q(s', a; \theta); \theta^-) \tag{6}$$

其中 $\arg\max_{a'} Q(s', a; \theta)$ 是目标网络最大输出Q值对应的动作，当前网络的预测值为 $Q'(s', \arg\max_{a'} Q(s', a; \theta); \theta^-)$。

---

## III. 路径规划综合方法

本节首先将AMR路径规划问题重新构建为MDP模型，MDP主要包含状态空间、动作空间和奖励函数。然后，提出一种改进的DDQN算法来求解AMR路径规划的MDP模型，并通过路径平滑方法获得最优路径。

### A. AMR路径规划的MDP模型

#### 1) 状态空间

基于网格的度量模型主要用于建立环境地图 [39], [40]。在此，三维环境的特征信息被映射到等大的二维网格中，如图5所示，使AMR能够在30 m × 30 m的二维网格环境中进行路径规划。根据环境中障碍物的分布，障碍物用黑色网格表示，AMR的自由移动区域（智能体的可行状态）用白色网格表示，每个网格的宽度为1 m。为便于构建仿真环境地图，未完全占据一个网格的障碍物将被视为一个完整的障碍物网格。

> **图5.** 环境模型示例（左：三维环境；右：二维网格环境）。（展示了将真实三维障碍物场景映射为二维网格地图的过程。）

由于在计算路径长度时需要考虑AMR与目标之间的欧氏距离 $TD_i$ ($i = 1, 2, \ldots, t$)，因此将AMR的位置坐标 $(x, y)$ 和 $TD_i$ 引入状态空间。此外，由于避障对路径规划和评估运动安全性至关重要，将传感器识别到的AMR与障碍物之间的欧氏距离定义为 $OD_i$ ($i = 1, 2, \ldots, t$)，如图6所示。$TD_i$ 和 $OD_i$ 均表示时刻 $i$ 在AMR周围一定距离内可获取的当前环境信息。

> **图6.** 状态空间。（展示了AMR周围的传感器检测范围，包括与目标的距离TD和与障碍物的距离OD。）

AMR的状态空间方程可以表示为：

$$S = \begin{bmatrix} x_1 & y_1 & TD_1 & OD_1 \\ x_2 & y_2 & TD_2 & OD_2 \\ \vdots & \vdots & \vdots & \vdots \\ x_t & y_t & TD_t & OD_t \end{bmatrix}_{t \times 4} \tag{7}$$

#### 2) 动作空间

一般而言，AMR的运动状态包括启动、停止、直线运动和转向，由伺服电机控制，路径规划过程中假定AMR的速度保持恒定。为简化AMR的运动模型，AMR的动作空间可描述为八个动作，即 $A = \{$上、下、左、右、右上、右下、左上、左下$\}$，具体动作空间如图7所示。

> **图7.** AMR动作空间。（展示了AMR在网格环境中的八个运动方向。）

基于此，AMR在网格环境中的转移规则可以表示为：

$$\begin{cases} (x', y') = (x + grid, y) \\ (x', y') = (x, y + grid) \\ (x', y') = (x - grid, y) \\ (x', y') = (x, y - grid) \\ (x', y') = (x + grid, y + grid) \\ (x', y') = (x + grid, y - grid) \\ (x', y') = (x - grid, y + grid) \\ (x', y') = (x - grid, y - grid) \end{cases} \tag{8}$$

其中 $(x, y)$ 表示AMR的当前状态，$(x', y')$ 表示AMR的下一状态，$grid$ 表示一个单元网格的宽度。需要强调的是，AMR在网格环境中每次执行的动作只能将其移动到相邻状态。

#### 3) 奖励函数设计

在MDP框架中，奖励函数用于评估智能体（AMR）动作 $a$ 的价值。在以往大多数研究中，当接近目标区域时，AMR获得正奖励；当与障碍物碰撞时，AMR获得负奖励。然而，其他状态的奖励值被设置为零，这导致了RL算法中的稀疏奖励（Sparse Reward）问题。因此，本文引入复合奖励函数 $R_t$，使智能体在算法训练过程中随着状态转移获得变化的奖励。复合奖励函数可以表示为：

$$R_t = r_{\tau_1} + r_{\tau_2} + r_{\tau_3} + r_{\tau_4} \tag{9}$$

其中 $r_{\tau_1}$ 是目标奖励函数，$r_{\tau_2}$ 是距离奖励函数，$r_{\tau_3}$ 是边界奖励函数，$r_{\tau_4}$ 是障碍物奖励函数。

**目标奖励函数** $r_{\tau_1}$ 通常设置为正值，用于激励AMR接近目标点（TP），计算公式为：

$$r_{\tau_1} = \lambda_1 \left(TD \leq \frac{\sqrt{2}}{2} grid\right) \tag{10}$$

其中 $\lambda_1$ 表示AMR到达目标区域时可获得的正奖励值。为提高MDP模型的训练效率，当AMR到达TP周围的黄色区域时即视为任务完成，如图8所示。

> **图8.** 目标区域示意图。（展示了目标点周围的到达判定区域。）

**距离奖励函数** $r_{\tau_2}$ 以欧氏距离作为启发式函数项 [41]，可以降低DDQN算法在环境搜索过程中的盲目性并提高规划效率，公式为：

$$r_{\tau_2} = \lambda_2 \sqrt{(x_k - x_{target})^2 + (y_k - y_{target})^2} \tag{11}$$

其中 $\lambda_2$ 是用于控制距离奖励函数幅度的负常数，$(x_k, y_k)$ 表示时刻 $k$ AMR的状态，$(x_{target}, y_{target})$ 表示目标状态。这意味着如果AMR越接近目标区域，将获得越小的负奖励值。因此，$r_{\tau_2}$ 能够促进AMR快速到达目标区域。

**边界奖励函数** $r_{\tau_3}$ 可以将AMR的运动限制在环境边界内，其定义为：

$$r_{\tau_3} = \begin{cases} \lambda_3, & x > \max(x) \text{ or } x < \min(x) \\ & y > \max(y) \text{ or } y < \min(y) \\ 0, & \text{otherwise} \end{cases} \tag{12}$$

式(12)中，$\lambda_3$ 表示一个负常数值，$\max(x)$ 和 $\min(x)$ 分别是环境的最大和最小水平坐标，$\max(y)$ 和 $\min(y)$ 分别是环境的最大和最小垂直坐标。

由于AMR车体具有不可忽略的宽度，**障碍物奖励函数** $r_{\tau_4}$ 可以有效避免AMR在实际运动过程中与障碍物边缘碰撞的风险。$r_{\tau_4}$ 的表达式为：

$$r_{\tau_4} = \begin{cases} \lambda_4, & OD < R_s \\ 0, & OD \geq R_s \end{cases} \tag{13}$$

其中 $\lambda_4$ 表示一个负值，AMR与障碍物之间的最小安全距离 $OD$ 设为 $R_s$，以半径 $R_s$ 的圆形区域作为最小安全区域，详细信息如图9所示。当AMR与障碍物之间的距离小于 $R_s$ ($R_s < 0.6$ m) 时，$r_{\tau_4}$ 给予AMR负奖励值 $\lambda_4$；当障碍物位于AMR最小安全区域之外 ($R_s \geq 0.6$ m) 时，不存在碰撞风险，奖励函数 $r_{\tau_4}$ 的值为0。

> **图9.** 最小安全距离示意图。（展示了AMR周围以 $R_s$ 为半径的最小安全区域及其与障碍物的距离关系。）

### B. AMR路径规划算法

#### 1) 深度神经网络

如图10所示，DDQN算法架构采用DNN来近似状态-动作价值函数 [42]。当前网络和目标网络均采用相同的DNN结构。DNN架构由一个输入层、三个隐藏层和一个输出层组成，网络参数以监督方式学习。神经网络的输入数据包括两部分：1）传感器扫描的环境信息；2）AMR的位置特征信息。这些数据通过全连接层处理，计算结果传输到输出层。输出层提供与输出动作集对应的Q值，最终输出最大Q值对应的动作。

> **图10.** 当前网络和目标网络的神经网络结构。（展示了包含输入层、三个隐藏层和输出层的DNN架构。）

在该网络架构中，修正线性单元（Rectified Linear Unit, ReLU）被用作输入层和每个隐藏层的激活函数。ReLU函数可表示为：

$$h(\xi) = \begin{cases} \xi, & \xi > 0 \\ 0, & \xi \leq 0 \end{cases} \tag{14}$$

为使DNN保持更好的学习效率，训练开始时学习率设为0.0025。然后利用自适应矩估计（Adam）算法优化学习率，实现学习率的自适应更新 [43]。同时，Adam引入动量项来改善神经网络的随机梯度下降。随后，当前网络结合防止梯度爆炸方法进行优化，从而提高DDQN算法的预测能力。

#### 2) 自适应动作选择策略

探索与利用之间的平衡是RL算法面临的一个挑战。智能体（AMR）需要充分利用已学习的经验，同时不断探索环境以获得更有效的长期动作策略。在先前的研究中，一种常见的方法是使用固定的 $\varepsilon$-贪心策略 [44]，AMR以一定概率选择经验中最大动作价值函数对应的动作或随机选择一个动作。然而，过度依赖历史经验可能导致AMR陷入局部最优，无法获得全局最优解；反之，过多的探索可能降低AMR的训练效率。

为在探索与利用之间取得平衡，本文提出了一种基于概率的非线性自适应 $\varepsilon$-贪心动作选择策略，其中概率 $\varepsilon$ 随训练回合（Episode）的变化进行自适应调整。该策略可以引导AMR在训练早期广泛探索环境并获取足够数量的训练样本；在训练后期，DDQN算法充分利用最优策略以加速算法收敛。自适应动作选择策略的表达式为：

$$\varepsilon_k = \varepsilon_f + \frac{\varepsilon_i - \varepsilon_f}{1 + e^{\frac{k}{\varepsilon_d}}} \tag{15}$$

其中 $\varepsilon_k$ 表示第 $k$ 回合的贪心因子，$\varepsilon_i$ 表示初始贪心因子，$\varepsilon_f$ 表示最终贪心因子，$\varepsilon_d$ 表示 $\varepsilon$-贪心的衰减速率。

**算法1：自适应 $\varepsilon$-贪心策略**

1. 生成随机概率值 $p$，$p \in (0, 1)$
2. 若 $p < \varepsilon_k$，则 $a = $ 随机动作
3. 否则若 $p \geq \varepsilon_k$，则 $a = \arg\max_a Q(s, a)$

#### 3) 路径平滑

DDQN算法的输出动作是离散的，而AMR在实际场景中的转向操作是连续的。为使规划路径更贴合AMR的实际需求，需要使用贝塞尔曲线对规划路径进行平滑处理，因为贝塞尔曲线具有编程难度低、计算成本低和轨迹连续性强等优点 [45], [46]。因此，将贝塞尔曲线引入DDQN算法框架，实现端到端的平滑路径规划。一阶到三阶贝塞尔曲线的计算表达式为：

$$p_i^k = \begin{cases} p_i^0, & i = 0, 1, 2, \ldots, n-k \\ (1-t)p_i^{k-1} + t \cdot p_{i+1}^{k-1}, & i = 0, 1, 2, \ldots, n-k \end{cases} \tag{16}$$

其中 $p_i^0$ 表示第 $i$ 个初始控制点，$p_i^k$ 表示第 $i$ 个 $K$ 阶控制点（$k = 1, 2, \ldots, n$），$t$ 表示比例系数。

根据式(16)，$n$ 阶控制点的计算公式总是包含两个 $n-1$ 阶控制点，因此 $n$ 阶控制点的计算公式最终可以从初始控制点计算得出。二阶贝塞尔曲线既能保证路径的平滑性，又比高阶贝塞尔曲线有更快的计算速度。因此，通过拼接多段二阶贝塞尔曲线来平滑路径的转折点，计算表达式为：

$$B(t) = (1-t)^2 p_i^0 + 2(1-t) p_{i+1}^0 + t^2 p_{i+2}^0, \quad t \in [0, 1] \tag{17}$$

其中 $p_i^0$、$p_{i+1}^0$ 和 $p_{i+2}^0$ 表示三个连续的初始化控制点。

#### 4) AMR路径规划算法流程

综上所述，IDDQN算法应用于AMR路径规划的流程图如图11所示。算法的伪代码如算法2所示，图11中连接线上的数字对应算法2中的流程。

> **图11.** 基于改进DDQN的AMR路径规划算法流程图。（展示了从参数初始化、环境交互、经验回放、网络更新到路径平滑输出的完整算法流程。）

**算法2：基于IDDQN的AMR路径规划算法**

1. **输入：** 交互次数 $N$；动作集 $A$；学习率 $\alpha$；折扣因子 $\gamma$；经验回放池最大容量 $D$；小批量 $M$；当前神经网络Q值 $Q$；目标神经网络Q值 $Q'$
2. 初始化网络参数 $\theta$ 和 $\theta'$，令 $\theta' = \theta$，并初始化回放池 $R$
3. 设置最大步数 $n_{max}$
4. **For** episode = 1, **do**
5. 初始化环境空间并获取起始状态 $s = (x_0, y_0)$
6. **For** step = 1, **do**
7. 将当前状态 $S$ 作为输入，获取当前状态对应的所有Q值输出，使用自适应 $\varepsilon$-贪心策略选择Q值对应的动作：以概率 $1-\varepsilon$ 选择 $a_t = \arg\max_{a'} Q(s, a; \theta)$，否则选择随机动作
8. AMR执行动作 $a$，与外部环境交互获得奖励 $r$ 和下一状态 $s'$
9. 将 $(s, a, r, s')$ 存入经验回放池
10. 令 $s = s'$
11. 从 $D$ 中随机采样小批量 $R$ 的转移 $(s_i, a_i, r_i, s_i')$，计算目标Q值 $y_i$
12. 设 $y_i = \begin{cases} r_i & \text{若回合在步骤 } i \text{ 终止} \\ r_i + \gamma Q'(S', \arg\max_{a'} Q(s', a'; \theta); \theta^-) & \text{否则} \end{cases}$
13. 计算损失函数 $L(\theta) = E[(y_i - Q(s, a; \theta))^2]$，通过神经网络梯度方向反向传播更新参数 $\theta$
14. 替换目标参数：$\theta^- = \tau\theta + (1-\tau)\theta^-$
15. 若 $s'$ 为目标状态，则当前交互结束，使用贝塞尔曲线平滑路径；否则进入步骤7
16. **输出：** 最优路径向量集
17. **End for**
18. **End for**

算法流程大致描述如下：

- **步骤1：** 设置训练参数，主要包括状态空间、动作集、初始学习率等；建立当前神经网络和目标神经网络；随机初始化参数 $\theta$ 和 $\theta^-$，且 $\theta = \theta^-$（第2行）。
- **步骤2：** AMR探索环境的过程（第5-8行）。在每个回合开始时初始化环境信息，在探索过程中将当前状态输入当前网络并输出相关动作的Q值（第5和第6行）。使用改进的自适应 $\varepsilon$-贪心策略获取并执行动作 $a$（第7行）。AMR与环境交互获得奖励 $r$ 和下一状态 $s'$（第8行）。
- **步骤3：** 当前网络和目标神经网络的更新过程（第9-14行）。首先将探索过程中获得的多样化样本数据保存到经验回放数据池中，然后从经验池中随机采样小批量经验数据更新当前网络（第9-11行）。更新目标Q值（第12行）。基于神经网络梯度的反向传播计算损失函数并更新参数 $\theta$ 的当前值（第13行）。执行目标网络的软更新（第14行）。
- **步骤4：** 经过多个回合后，保存最终神经网络参数，AMR输出最优路径（第15行）。

---

## IV. 仿真研究

为验证所提IDDQN算法的可行性，在PyCharm平台上建立了IDDQN及其他四种路径规划算法的仿真模型，并利用gym框架设计了四个30 × 30维的可视化网格仿真环境。IDDQN算法采用的超参数配置汇总于表I。训练算法所用的操作系统设备为Windows 11，CPU为i7-12700H，GPU为NVIDIA RTX 3050 TI。在相同的四个未知障碍物环境中，将IDDQN算法与基线算法进行比较，以验证IDDQN算法的有效性和优越性。同时，在具有随机干扰的未知环境中初步评估了IDDQN算法的鲁棒性。

> **表I.** 超参数设置。（列出了IDDQN算法训练所用的关键超参数。）

### A. 基线算法

用于对比的算法描述如下：

1. **IDDQN（本文提出）：** 基于DRL，联合优化动作选择策略、DNN和奖励函数。通过引入综合奖励函数，提高了AMR的路径规划安全性和搜索效率。
2. **APF [10]：** APF算法通过计算环境中势场的梯度，能够快速灵活地规划有效路径，路径轨迹可与障碍物保持安全距离。
3. **RRT [47]：** RRT算法通过不断扩展随机生成的树来探索解空间，能够适应各种环境并避免局部最优解，同时实现高效的实时路径规划。
4. **A\* [48]：** A\*算法利用广度优先搜索和最佳优先搜索原理精确引导机器人进行环境探索，并确保在有限状态空间内找到最短路径。

根据APF算法、RRT算法和A\*算法的优势，从四个方面评估不同算法规划的路径结果的性能：1）路径长度；2）路径平滑性；3）路径轨迹与障碍物之间的距离；4）算法推理时间（Inference Time, IT），以评估所提IDDQN算法的优越性和有效性。

此外，所有对比算法的计算复杂度列举如下：IDDQN算法的计算复杂度为 $O(EP \times N)$，与传统DQN算法的时间复杂度相似，其中 $N$ 为每个回合的时间步数，$EP$ 为回合数。A\*算法的时间复杂度可表示为 $O(N_1 \log N_1)$，其中 $N_1$ 为节点数。APF算法的时间复杂度表达式为 $O(N_2 \times M)$，其中 $N_2$ 为迭代次数，$M$ 为障碍物数量。RRT算法的时间复杂度为 $O(N_3^2)$，其中 $N_3$ 为树的顶点数。

### B. 无干扰未知环境下的仿真

在此，设计了四种具有不同地形特征和障碍物密度的未知环境，以评估所提IDDQN算法的性能。首先建立了五个关键性能指标（Key Performance Indicators, KPIs）来评估各种路径规划算法的性能。KPIs包括路径拐角数（Path Corners, PCs）、最大路径转弯角度（Maximum Path Turning Angle, MPTA）、最小碰撞距离（Minimum Distance To Collision, MDTC）、平均路径长度（Average Path Length, APL）以及算法推理时间（IT）。五种算法在四个不同仿真环境下的规划路径如图12所示，KPIs值如表II所示。

> **图12.** 不同路径规划方法的仿真结果。（展示了(a)-(d)四种不同障碍物环境下IDDQN、DQN、APF、RRT和A\*五种算法的路径规划结果对比。）

> **表II.** 不同路径规划算法的KPIs对比。（包含四种环境下各算法的PCs、MPTA、MDTC、APL和IT数据。）

如图12所示，五种路径规划算法均能在四种不同的未知环境中成功生成无碰撞路径。一方面，基于APF、RRT和A\*三种常用传统路径规划方法的规划轨迹以及表II中KPIs的对比分析可以观察到：APF算法能够在保证运动安全的前提下实时计算AMR运动轨迹，但APF算法容易陷入局部极小值，可能导致生成路径中出现局部振荡和更长的路径长度。RRT算法在路径规划中表现出较高的搜索效率，但生成的路径长度较长、路径转折点较多，且路径轨迹更靠近障碍物，无法保证AMR路径规划的安全性。此外，虽然A\*算法规划的路径相对较短且转折点较少，但A\*算法生成的轨迹存在与障碍物边缘碰撞的风险。而且，随着环境复杂度的增加，A\*算法的推理时间将显著增加。

另一方面，通过比较DQN和IDDQN算法在四种未知复杂环境下的路径规划性能结果，可以得出：DQN和IDDQN算法生成的轨迹能够与障碍物保持至少0.5 m的距离，总体上优于三种传统路径规划算法。更重要的是，与DQN算法相比，IDDQN算法在四种复杂环境下的APL分别降低了11.69%、8.49%、7.26%和9.82%，且IDDQN算法生成的规划路径具有更好的平滑性和更少的路径转折点。

图13展示了DQN和IDDQN算法在四种不同环境下训练过程中奖励值的变化。由于IDDQN算法在训练早期阶段智能体（AMR）对环境进行了大量探索，IDDQN算法的奖励函数曲线出现了明显的振荡。随后，AMR利用从探索中获得的经验知识加速学习过程，最终使学习奖励趋于稳定。随着障碍物环境复杂度的增加，IDDQN和DQN都需要更多的探索步骤才能使奖励值稳定。虽然DQN算法在经过一段时间的持续学习和试错后可以实现奖励曲线的稳定，但由于Q值的过估计和过多的无效动作选择策略，其收敛速度较慢。此外，奖励曲线在训练后期表现出显著波动。

> **图13.** IDDQN和DQN训练过程中奖励值的变化。（展示了(a)-(d)四种环境下两种算法的训练奖励曲线对比。）

通过比较IDDQN和DQN的奖励值变化趋势，可以看出在四种不同环境下，IDDQN算法达到稳定奖励值所需的迭代步数分别比DQN算法减少了约27.31%、32.17%、52.91%和26.40%。对比结果表明IDDQN具有更高的学习效率、全局搜索能力和优异的收敛性。

图14所示的雷达图基于表II中记录的数据生成，便于分析IDDQN算法与其他四种算法之间路径规划性能的差异。从图14可以看出，在四种实验环境下，IDDQN算法模型的推理速度远优于APF、RRT和A\*算法。IDDQN算法生成路径的路径长度、转折点数量和最大转弯角度均小于DQN算法和其他传统方法。因此，IDDQN算法采用端到端训练模式，具有更好的综合路径规划能力。

> **图14.** 四种环境下不同路径规划算法KPIs的对比结果。（展示了五种算法在多个KPI维度上的雷达图对比。）

### C. 有干扰未知环境下的仿真

AMR在运动过程中可能会经历外部环境因素的变化，如路面摩擦力不足和地面条件不均匀，这可能导致AMR的稳定性变差和方向偏移。因此，AMR的运动将由动作选择策略和外部环境信息共同决定。具体而言，位移干扰对AMR路径规划的影响可以描述为：AMR在经过随机干扰位置 $(x, y)_{current}$ 后到达下一状态 $(x, y)_{next}$ 时，将在 $x$ 轴或 $y$ 轴方向产生随机偏移 $(a, b)$，其中 $(x, y)_{next} = (x, y)_{current} + (a, b)$。

为评估所提IDDQN算法在具有随机干扰的未知环境中的适应性和鲁棒性，在两个20 × 20单目标复杂障碍物环境和两个30 × 30多目标复杂障碍物环境中随机设置了多种类型的干扰。单目标环境中随机干扰的位置和大小记录在表III中。在多目标环境中，AMR的路径规划过程由各目标区域划分为不同阶段，AMR各运行阶段的随机干扰位置和大小记录在表IV中。

> **表III.** 单目标障碍物环境中的随机干扰分布和大小。
>
> **表IV.** 多目标障碍物环境中的随机干扰分布和大小。

IDDQN算法在有干扰和无干扰训练环境中的路径规划结果如图15所示。

> **图15.** (a)和(b) AMR在单目标未知干扰环境中的轨迹；(c)和(d) AMR在多目标未知干扰环境中的轨迹。（展示了有无干扰条件下路径偏差的对比。）

同时，图16给出了五个KPIs来评价环境干扰对路径规划结果的影响，包括APL、达到稳定奖励值所需的总探索步数（Total Steps, TS）、最小碰撞距离（MDTC）、路径拐角数（PCs）和最大路径转弯角度（MPTA）。

> **图16.** 不同环境中干扰对AMR路径规划的影响。（展示了有无干扰条件下各KPI指标的对比柱状图。）

从图15可以观察到，无论是在单目标环境还是多目标环境中，无论是否存在干扰，IDDQN算法都能使AMR避开障碍物到达各个目标区域。然而，环境干扰的存在导致IDDQN算法规划的路径与无干扰环境中规划的路径相比出现了偏差。此外，环境中的干扰信息越多、连续目标点越多，路径偏差就越显著。在环境干扰条件下，IDDQN生成的最终路径更长且包含更多的路径转折点。

通过比较图15中的KPIs数据可以发现，在两种不同环境中，IDDQN算法的路径长度和总探索步数受环境干扰影响显著。特别是根据图15(d)，在有干扰的未知环境中获得的路径长度比原始路径增加了10.12%，IDDQN算法达到稳定奖励值所需的探索步数增加了66.06%。此外，IDDQN算法需要在避障的同时减轻环境中随机干扰的影响，导致生成的路径中转弯角度更大、路径转折点更多。仿真实验表明，环境中的随机干扰并不妨碍IDDQN算法为AMR规划到达所有目标区域的最优路径，同时确保路径轨迹与不同障碍物保持最小安全距离。但环境中干扰的存在会增加IDDQN算法的计算需求。

总之，IDDQN算法将在未知环境中从零开始持续学习干扰信息，并能展现出一定水平的适应性和鲁棒性。所提出的IDDQN算法模型可以扩展应用于更多环境，为AMR路径规划提供了新的解决方案。

---

## V. 结论

本文提出一种改进的DDQN算法来处理综合未知环境中AMR的路径规划问题。首先，在建立AMR路径规划MDP模型后，设计了综合奖励函数，以快速引导AMR到达目标区域，同时确保AMR与障碍物保持安全距离。然后，在DDQN算法框架中，融合了自适应 $\varepsilon$-贪心策略和优化的DNN，提高了DDQN算法的全局搜索能力和收敛速度。此外，引入贝塞尔曲线算法对DDQN输出的离散动作进行平滑处理。最后，在各种综合未知环境中验证了IDDQN算法。

在四组未知环境中，与DQN、A\*、RRT和APF算法相比，所提IDDQN算法不仅能为AMR规划更安全、更短的路径，而且其推理速度远优于传统方法。同时，通过评估IDDQN算法在具有随机干扰的未知环境中的性能（包括单目标和多目标场景），进一步证明了IDDQN算法具有更好的适应性和鲁棒性。

对于未来工作，我们将聚焦于基于传感器的动态路径规划，并进行硬件在环实验的实际算法验证。

---

## 参考文献

[1] A. Loganathan and N. S. Ahmad, "A systematic review on recent advances in autonomous mobile robot navigation," Eng. Sci. Techno., Int. J., vol. 40, Apr. 2023, Art. no. 101343.

[2] F. H. Ajeil, I. K. Ibraheem, M. A. Sahib, and A. J. Humaidi, "Multi-objective path planning of an autonomous mobile robot using hybrid PSO-MFB optimization algorithm," Appl. Soft Comput., vol. 89, Apr. 2020, Art. no. 106076.

[3] J. Li, B. Zhao, and F. Xu, "Dynamic path planning of intelligent robot in power equipment maintenance environment," Energy Rep., vol. 9, pp. 784–791, Sep. 2023.

[4] X. Sun, S. Deng, B. Tong, S. Wang, C. Zhang, and Y. Jiang, "Hierarchical framework for mobile robots to effectively and autonomously explore unknown environments," ISA Trans., vol. 134, pp. 1–15, Mar. 2023.

[5] Y. Ma, Y. Zhao, Z. Li, X. Yan, H. Bi, and G. Królczyk, "A new coverage path planning algorithm for unmanned surface mapping vehicle based on A-star based searching," Appl. Ocean Res., vol. 123, Jun. 2022, Art. no. 103163.

[6] Y. Liu, P. Huang, F. Zhang, and Y. Zhao, "Distributed formation control using artificial potentials and neural network for constrained multiagent systems," IEEE Trans. Control Syst. Technol., vol. 28, no. 2, pp. 697–704, Mar. 2020.

[7] G. Ma, Y. Duan, M. Li, Z. Xie, and J. Zhu, "A probability smoothing Bi-RRT path planning algorithm for indoor robot," Future Gener. Comput. Syst., vol. 143, pp. 349–360, Jun. 2023.

[8] X. Yu, W.-N. Chen, T. Gu, H. Yuan, H. Zhang, and J. Zhang, "ACO-A*: Ant colony optimization plus A* for 3-D traveling in environments with dense obstacles," IEEE Trans. Evol. Comput., vol. 23, no. 4, pp. 617–631, Aug. 2019.

[9] L. Li, D. Wu, Y. Huang, and Z. M. Yuan, "A path planning strategy unified with a COLREGS collision avoidance function based on deep reinforcement learning and artificial potential field," Appl. Ocean Res., vol. 113, Aug. 2021, Art. no. 102759.

[10] Z. Lin, M. Yue, G. Chen, and J. Sun, "Path planning of mobile robot with PSO-based APF and fuzzy-based DWA subject to moving obstacles," Trans. Inst. Meas. Control, vol. 44, no. 1, pp. 121–132, 2022.

[11] J. Ding, Y. Zhou, X. Huang, K. Song, S. Lu, and L. Wang, "An improved RRT algorithm for robot path planning based on path expansion heuristic sampling," J. Comput. Sci., vol. 67, Mar. 2023, Art. no. 101937.

[12] Z. Yu, Z. Si, X. Li, D. Wang, and H. Song, "A novel hybrid particle swarm optimization algorithm for path planning of UAVs," IEEE Internet Things J., vol. 9, no. 22, pp. 22547–22558, Nov. 2022.

[13] C.-C. Tsai, H.-C. Huang, and C.-K. Chan, "Parallel elite genetic algorithm and its application to global path planning for autonomous robot navigation," IEEE Trans. Ind. Electron., vol. 58, no. 10, pp. 4813–4821, Oct. 2011.

[14] H. Yang, J. Qi, Y. Miao, H. Sun, and J. Li, "A new robot navigation algorithm based on a double-layer ant algorithm and trajectory optimization," IEEE Trans. Ind. Electron., vol. 66, no. 11, pp. 8557–8566, Nov. 2019.

[15] M. L. Littman, "Reinforcement learning improves behavior from evaluative feedback," Nature, vol. 521, no. 7553, pp. 445–451, 2015.

[16] R. Liu, M. Xie, A. Liu, and H. Song, "Joint optimization risk factor and energy consumption in IoT networks with tiny ML-enabled Internet of UAVs," IEEE Internet Things J., early access, Jan. 1, 2024, doi: 10.1109/JIOT.2023.3348837.

[17] M. Pflueger, A. Agha, and G. S. Sukhatme, "Rover-IRL: Inverse reinforcement learning with soft value iteration networks for planetary rover path planning," IEEE Robot. Autom. Lett., vol. 4, no. 2, pp. 1387–1394, Apr. 2019.

[18] B. Wang, Z. Liu, Q. Li, and A. Prorok, "Mobile robot path planning in dynamic environments through globally guided reinforcement learning," IEEE Robot. Autom. Lett., vol. 5, no. 4, pp. 6932–6939, Oct. 2020.

[19] Q. Zhou, Y. Lian, J. Wu, M. Zhu, H. Wang, and J. Cao, "An optimized Q-Learning algorithm for mobile robot local path planning," Knowl. Based Syst., vol. 286, Feb. 2024, Art. no. 111400.

[20] E. S. Low, P. Ong, and K. C. Cheah, "Solving the optimal path planning of a mobile robot using improved Q-learning," Robot. Auton. Syst., vol. 115, pp. 143–161, May 2019.

[21] M. Pei, H. An, B. Liu, and C. Wang, "An improved Dyna-Q algorithm for mobile robot path planning in unknown dynamic environment," IEEE Trans. Syst., Man, Cybern., Syst., vol. 52, no. 7, pp. 4415–4425, Jul. 2022.

[22] C. S. Tan, R. Mohd-Mokhtar, and M. R. Arshad, "Expected-mean gamma-incremental reinforcement learning algorithm for robot path planning," Expert Syst. Appl., vol. 249, Sep. 2024, Art. no. 123539.

[23] J. Yang, J. Ni, M. Xi, J. Wen, and Y. Li, "Intelligent path planning of underwater robot based on reinforcement learning," IEEE Trans. Autom. Sci. Eng., vol. 20, no. 3, pp. 1983–1996, Jul. 2023.

[24] Z. Chu, F. Wang, T. Lei, and C. Luo, "Path planning based on deep reinforcement learning for autonomous underwater vehicles under ocean current disturbance," IEEE Trans. Intell. Veh., vol. 8, no. 1, pp. 108–120, Jan. 2023.

[25] Y. Xiaofei, S. Yilun, L. Wei, Y. Hui, Z. Weibo, and X. Zhengrong, "Global path planning algorithm based on double DQN for multi-tasks amphibious unmanned surface vehicle," Ocean Eng., vol. 266, Dec. 2022, Art. no. 112809.

[26] L. Yu, F. Wu, Z. Xu, Z. Xie, and D. Yang, "UAV path design with connectivity constraint based on deep reinforcement learning," Phys. Commun., vol. 52, Jun. 2022, Art. no. 101582.

[27] K. Zheng, H. Yang, S. Liu, K. Zhang, and L. Lei, "A behavior decision method based on reinforcement learning for autonomous driving," IEEE Internet Things J., vol. 9, no. 24, pp. 25386–25394, Dec. 2022.

[28] S. Wen, Y. Zhao, X. Yuan, Z. Wang, D. Zhang, and L. Manfredi, "Path planning for active SLAM based on deep reinforcement learning under unknown environments," Intell. Service Robot., vol. 13, pp. 263–272, 2022.

[29] K. Wu, H. Wang, M. A. Esfahani, and S. Yuan, "BND*-DDQN: Learn to steer autonomously through deep reinforcement learning," IEEE Trans. Cogn. Develop. Syst., vol. 13, no. 2, pp. 249–261, Jun. 2021.

[30] X. Wang, H. Fu, G. Deng, C. Liu, K. Tang, and C. Chen, "Hierarchical free gait motion planning for hexapod robots using deep reinforcement learning," IEEE Trans. Ind. Informat., vol. 19, no. 11, pp. 10901–10912, Nov. 2023.

[31] X. Tao and A. S. Hafid, "DeepSensing: A novel mobile crowdsensing framework with double deep q-network and prioritized experience replay," IEEE Internet Things J., vol. 7, no. 12, pp. 11547–11558, Dec. 2020.

[32] J. Jiang, X. Zeng, D. Guzzetti, and Y. You, "Path planning for asteroid hopping rovers with pre-trained deep reinforcement learning architectures," Acta Astronautica, vol. 171, pp. 265–279, Jun. 2020.

[33] W. Lan et al., "Path planning for underwater gliders in time-varying ocean current using deep reinforcement learning," Ocean Eng., vol. 262, Oct. 2022, Art. no. 112226.

[34] M. Xi, J. Yang, J. Wen, H. Liu, Y. Li, and H. H. Song, "Comprehensive ocean information-enabled AUV path planning via reinforcement learning," IEEE Internet Things J., vol. 9, no. 18, pp. 17440–17451, Sep. 2022.

[35] W. Zhang, J. Gai, Z. Zhang, L. Tang, Q. Liao, and Y. Ding, "Double-DQN based path smoothing and tracking control method for robotic vehicle navigation," Comput. Electron. Agric., vol. 166, Nov. 2019, Art. no. 104985.

[36] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[37] S. Wen, Z. Wen, D. Zhang, H. Zhang, and T. Wang, "A multi-robot path-planning algorithm for autonomous navigation using meta-reinforcement learning based on transfer learning," Appl. Soft Comput., vol. 110, Oct. 2021, Art. no. 107605.

[38] H. Van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double q-learning," in Proc. AAAI Conf. Artif. Intell., 2016, pp. 2094–2100.

[39] A. Koval, S. Karlsson, and G. Nikolakopoulos, "Experimental evaluation of autonomous map-based spot navigation in confined environments," Biomim. Intell. Robot., vol. 2, no. 1, 2022, Art. no. 100035.

[40] Y. Wu, F. Xie, L. Huang, R. Sun, J. Yang, and Q. Yu, "Convolutionally evaluated gradient first search path planning algorithm without prior global maps," Robot. Auton. Syst., vol. 150, Apr. 2022, Art. no. 103985.

[41] J. Wen, J. Yang, and T. Wang, "Path planning for autonomous underwater vehicles under the influence of ocean currents based on a fusion heuristic algorithm," IEEE Trans. Veh. Technol., vol. 70, no. 9, pp. 8529–8544, Sep. 2021.

[42] B. Hadi, A. Khosravi, and P. Sarhadi, "Deep reinforcement learning for adaptive path planning and control of an autonomous underwater vehicle," Appl. Ocean Res., vol. 129, Dec. 2022, Art. no. 103326.

[43] G. Chen, Z. Gao, M. Hua, B. Shuai, and Z. Gao, "Lane change trajectory prediction considering driving style uncertainty for autonomous vehicles," Mech. Syst. Signal Process., vol. 206, Jan. 2024, Art. no. 110854.

[44] A. Bozanta et al., "Courier routing and assignment for food delivery service using reinforcement learning," Comput. Ind. Eng., vol. 164, Feb. 2022, Art. no. 107871.

[45] B. Song, Z. Wang, and L. Zou, "An improved PSO algorithm for smooth path planning of mobile robots using continuous high-degree Bezier curve," Appl. Soft Comput., vol. 100, Mar. 2021, Art. no. 106960.

[46] G. Klancar and S. Blazic, "Optimal constant acceleration motion primitives," IEEE Trans. Veh. Technol., vol. 68, no. 9, pp. 8502–8511, Sep. 2019.

[47] C.-b. Moon and W. Chung, "Kinodynamic planner dual-tree RRT (DT-RRT) for two-wheeled mobile robots using the rapidly exploring random tree," IEEE Trans. Ind. Electron., vol. 62, no. 2, pp. 1080–1090, Feb. 2015.

[48] B. Fu et al., "An improved A* algorithm for the industrial robot path planning with high success rate and short length," Robot. Auton. Syst., vol. 106, pp. 26–37, Aug. 2018.

---

## 作者简介

**Zekun Bai** 于2021年获得陕西理工大学车辆工程学士学位，目前在西安理工大学机械与精密仪器工程学院攻读机械工程硕士学位。主要研究方向包括强化学习、路径规划和车辆智能控制。

**Hui Pang**（IEEE会员）于2002年获得郑州航空工业管理学院机械制造及其自动化学士学位，2005年和2009年分别获得西北工业大学机械工程硕士和博士学位。现为西安理工大学机械与精密仪器工程学院教授。2016年11月至2017年11月在美国克莱姆森大学汽车工程系担任访问学者。研究方向包括非线性鲁棒控制和模糊滑模控制及其在车辆动力学控制中的应用。

**Zhaonian He** 于2022年获得聊城大学机械工程学士学位，目前在西安理工大学机械与精密仪器工程学院攻读机械工程硕士学位。主要研究方向包括无人车辆的路径规划与控制。

**Bin Zhao** 于2023年获得黑龙江工程学院车辆工程学士学位，目前在西安理工大学机械与精密仪器工程学院攻读机械工程硕士学位。主要研究方向包括深度强化学习、路径规划和车辆智能控制。

**Tong Wang** 于2023年获得西安理工大学机械设计制造及其自动化学士学位，目前在该校机械与精密仪器工程学院攻读机械工程硕士学位。主要研究方向包括深度强化学习、智能车辆轨迹跟踪和路径规划。
