# 用于深度强化学习的对决网络架构

**Ziyu Wang**, **Tom Schaul**, **Matteo Hessel**, **Hado van Hasselt**, **Marc Lanctot**, **Nando de Freitas**

Google DeepMind, London, UK

arXiv:1511.06581v3 [cs.LG] 5 Apr 2016

---

## 摘要

近年来，深度表征（Deep Representation）在强化学习（Reinforcement Learning, RL）中取得了诸多成功。然而，大多数应用仍然采用传统架构，如卷积网络（Convolutional Network）、长短期记忆网络（LSTM）或自编码器（Auto-encoder）。本文提出了一种用于无模型强化学习（Model-free Reinforcement Learning）的新型神经网络架构。我们的对决网络（Dueling Network）包含两个独立的估计器：一个用于状态价值函数（State Value Function），另一个用于状态依赖的动作优势函数（State-dependent Action Advantage Function）。这种分解的主要优势在于，无需对底层强化学习算法做任何改变，即可实现跨动作的学习泛化。实验结果表明，该架构在存在许多相似值动作的情况下能够更好地进行策略评估。此外，对决架构使我们的 RL 智能体在 Atari 2600 域上超越了当时的最先进方法。

---

## 1. 引言

过去几年，深度学习为机器学习的可扩展性和性能带来了巨大进步 (LeCun et al., 2015)。其中一个令人振奋的应用是强化学习和控制中的序贯决策（Sequential Decision-making）场景。著名的例子包括深度 Q 学习 (Mnih et al., 2015)、深度视觉运动策略 (Levine et al., 2015)、基于注意力的循环网络 (Ba et al., 2015) 以及带嵌入的模型预测控制 (Watter et al., 2015)。其他近期成果包括大规模并行框架 (Nair et al., 2015) 和围棋中的专家走法预测 (Maddison et al., 2015)，后者产生的策略与蒙特卡洛树搜索（Monte Carlo Tree Search）程序相当，并在结合搜索后完胜职业棋手 (Silver et al., 2016)。

尽管如此，大多数 RL 方法仍使用标准神经网络，如卷积网络、多层感知机（MLP）、LSTM 和自编码器。近期进展的重点在于设计改进的控制和 RL 算法，或者仅仅是将现有神经网络架构纳入 RL 方法。本文采取了一种互补的替代方法，主要关注创新一种更适合无模型 RL 的神经网络架构。这种方法的优势在于，新网络可以很容易地与现有和未来的 RL 算法结合。也就是说，本文提出了一种新网络（图 1），但使用的是已发表的算法。

所提出的网络架构，我们称之为对决架构（Dueling Architecture），显式地将状态价值和（状态依赖的）动作优势的表示分离开来。对决架构由两个流（Stream）组成，分别表示价值函数和优势函数，同时共享一个公共的卷积特征学习模块。两个流通过一个特殊的聚合层（Aggregating Layer）合并，产生状态-动作价值函数 $Q$ 的估计，如图 1 所示。该对决网络应被理解为一个具有两个流的单一 Q 网络，用于替代现有算法（如深度 Q 网络 DQN；Mnih et al., 2015）中流行的单流 Q 网络。对决网络自动产生状态价值函数和优势函数的独立估计，无需任何额外监督。

直观地说，对决架构能够学习哪些状态是（或不是）有价值的，而无需学习每个动作在每个状态下的效果。这在动作不以任何相关方式影响环境的状态中特别有用。为说明这一点，可参考图 2 所示的显著性图（Saliency Map）。这些图是通过计算训练好的价值流和优势流相对于输入视频的雅可比矩阵（Jacobian）生成的，遵循 Simonyan et al. (2013) 提出的方法。图中展示了两个不同时间步的价值和优势显著性图。在一个时间步中（最左边的一对图像），价值网络流关注道路，特别是地平线上新出现的汽车，同时也关注分数。另一方面，优势流并不太关注视觉输入，因为当前方没有汽车时，其动作选择实际上是无关紧要的。然而，在第二个时间步中（最右边的一对图像），优势流则密切关注，因为前方有一辆车迫在眉睫，使得其动作选择非常关键。

在实验中，我们证明了当冗余或相似的动作被添加到学习问题中时，对决架构能够更快地识别正确的动作。

我们还在极具挑战性的 Atari 2600 测试平台上评估了对决架构带来的收益。在该平台上，一个具有相同结构和超参数的 RL 智能体必须仅通过观察图像像素和游戏分数来学会玩 57 种不同的游戏。结果表明，相比 Mnih et al. (2015) 和 van Hasselt et al. (2015) 的单流基线，对决架构取得了巨大的改进。优先回放（Prioritized Replay）(Schaul et al., 2016) 与所提出的对决网络的结合，在该热门测试域上取得了新的最先进成果。

> **图 1 描述**：上方为流行的单流 Q 网络，下方为对决 Q 网络。对决网络有两个流，分别估计（标量）状态价值和每个动作的优势；绿色输出模块实现公式 (9) 将二者合并。两种网络都为每个动作输出 Q 值。

> **图 2 描述**：Atari 游戏 Enduro 上训练好的对决架构的价值和优势显著性图（红色叠加层）。价值流学会关注道路。优势流仅在前方有车辆时才关注，以避免碰撞。

### 1.1 相关工作

维护独立价值函数和优势函数的概念可以追溯到 Baird (1993)。在 Baird 最初的优势更新（Advantage Updating）算法中，共享的贝尔曼残差（Bellman Residual）更新方程被分解为两个更新：一个用于状态价值函数，一个用于其关联的优势函数。在简单的连续时间域上，优势更新被证明比 Q 学习收敛更快 (Harmon et al., 1995)。其后继者——优势学习（Advantage Learning）算法——仅表示单个优势函数 (Harmon & Baird, 1996)。

对决架构用一个深度模型同时表示价值 $V(s)$ 和优势 $A(s, a)$ 函数，其输出将两者结合以产生状态-动作价值 $Q(s, a)$。与优势更新不同的是，表示和算法在构造上是解耦的。因此，对决架构可以与众多无模型 RL 算法结合使用。

优势函数在策略梯度（Policy Gradient）中有着悠久的历史，始于 Sutton et al. (2000)。作为该研究方向的近期例子，Schulman et al. (2015) 在线估计优势值以降低策略梯度算法的方差。

已有多项使用深度强化学习玩 Atari 游戏的尝试，包括 Mnih et al. (2015)、Guo et al. (2014)、Stadie et al. (2015)、Nair et al. (2015)、van Hasselt et al. (2015)、Bellemare et al. (2016) 和 Schaul et al. (2016)。Schaul et al. (2016) 的结果是当时已发表的最先进水平。

---

## 2. 背景

我们考虑一个序贯决策设置，其中智能体在离散时间步上与环境 $\mathcal{E}$ 交互，可参见 Sutton & Barto (1998) 的介绍。以 Atari 域为例，智能体在时间步 $t$ 感知到由 $M$ 帧图像组成的视频 $s_t$：$s_t = (x_{t-M+1}, \ldots, x_t) \in \mathcal{S}$。然后智能体从离散动作集 $a_t \in \mathcal{A} = \{1, \ldots, |\mathcal{A}|\}$ 中选择一个动作，并观察由游戏模拟器产生的奖励信号 $r_t$。

智能体的目标是最大化期望折扣回报（Discounted Return），其中折扣回报定义为 $R_t = \sum_{\tau=t}^{\infty} \gamma^{\tau-t} r_\tau$。在此公式中，$\gamma \in [0, 1]$ 是折扣因子（Discount Factor），用于权衡即时奖励和未来奖励的重要性。

对于按照随机策略 $\pi$ 行动的智能体，状态-动作对 $(s, a)$ 的价值和状态 $s$ 的价值定义如下：

$$Q^\pi(s, a) = \mathbb{E}[R_t | s_t = s, a_t = a, \pi]$$

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(s)}[Q^\pi(s, a)]$$
$\hfill (1)$

上述状态-动作价值函数（简称 Q 函数）可通过动态规划（Dynamic Programming）递归计算：

$$Q^\pi(s, a) = \mathbb{E}_{s'}\left[r + \gamma \mathbb{E}_{a' \sim \pi(s')}[Q^\pi(s', a')] \mid s, a, \pi\right]$$

定义最优 $Q^*(s, a) = \max_\pi Q^\pi(s, a)$。在确定性策略 $a = \arg\max_{a' \in \mathcal{A}} Q^*(s, a')$ 下，有 $V^*(s) = \max_a Q^*(s, a)$。由此可得最优 Q 函数满足贝尔曼方程（Bellman Equation）：

$$Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$
$\hfill (2)$

我们定义另一个重要量——优势函数（Advantage Function），它关联了价值函数和 Q 函数：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$
$\hfill (3)$

注意 $\mathbb{E}_{a \sim \pi(s)}[A^\pi(s, a)] = 0$。直观地说，价值函数 $V$ 衡量处于特定状态 $s$ 的好坏程度。而 Q 函数衡量的是在该状态下选择特定动作的价值。优势函数从 Q 函数中减去状态价值，得到每个动作重要性的相对度量。

### 2.1 深度 Q 网络

前一节描述的价值函数是高维对象。为近似它们，我们可以使用参数为 $\theta$ 的深度 Q 网络：$Q(s, a; \theta)$。为估计该网络，我们在迭代 $i$ 处优化以下损失函数序列：

$$L_i(\theta_i) = \mathbb{E}_{s, a, r, s'}\left[\left(y_i^{DQN} - Q(s, a; \theta_i)\right)^2\right]$$
$\hfill (4)$

其中

$$y_i^{DQN} = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$
$\hfill (5)$

这里 $\theta^-$ 表示固定且独立的目标网络（Target Network）的参数。我们可以尝试使用标准 Q 学习在线学习网络 $Q(s, a; \theta)$ 的参数，但该估计器在实践中表现不佳。Mnih et al. (2015) 的一个关键创新是：在通过梯度下降更新在线网络 $Q(s, a; \theta_i)$ 时，将目标网络 $Q(s', a'; \theta^-)$ 的参数冻结固定次数的迭代。这大大提高了算法的稳定性。具体的梯度更新为：

$$\nabla_{\theta_i} L_i(\theta_i) = \mathbb{E}_{s, a, r, s'}\left[\left(y_i^{DQN} - Q(s, a; \theta_i)\right) \nabla_{\theta_i} Q(s, a; \theta_i)\right]$$

该方法是无模型的，因为状态和奖励由环境产生。它也是离策略的（Off-policy），因为这些状态和奖励是通过与正在学习的在线策略不同的行为策略（DQN 中为 $\epsilon$-贪心策略）获得的。

DQN 成功的另一个关键要素是经验回放（Experience Replay）(Lin, 1993; Mnih et al., 2015)。在学习过程中，智能体从多个回合中积累经验数据集 $D_t = \{e_1, e_2, \ldots, e_t\}$，其中 $e_t = (s_t, a_t, r_t, s_{t+1})$。训练 Q 网络时，不是像标准时序差分学习那样仅使用当前经验，而是从 $D$ 中均匀随机采样小批量经验来训练网络。损失函数序列因此变为：

$$L_i(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim U(D)}\left[\left(y_i^{DQN} - Q(s, a; \theta_i)\right)^2\right]$$

经验回放通过在多次更新中重用经验样本来提高数据效率，更重要的是，从回放缓冲区中均匀采样减少了更新中使用的样本之间的相关性，从而降低了方差。

### 2.2 双重深度 Q 网络

前一节描述了 Mnih et al. (2015) 中提出的 DQN 的主要组件。在本文中，我们使用 van Hasselt et al. (2015) 改进的双重 DQN（Double DQN, DDQN）学习算法。在 Q 学习和 DQN 中，max 算子使用相同的值来选择和评估动作，这可能导致过于乐观的值估计 (van Hasselt, 2010)。为缓解这一问题，DDQN 使用以下目标：

$$y_i^{DDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta_i); \theta^-)$$
$\hfill (6)$

DDQN 与 DQN 相同（参见 Mnih et al., 2015），但目标 $y_i^{DQN}$ 被替换为 $y_i^{DDQN}$。DDQN 的伪代码见附录 A。

### 2.3 优先回放

优先经验回放（Prioritized Experience Replay）(Schaul et al., 2016) 是在 DDQN 基础上的一项近期创新，进一步提升了最先进水平。其关键思想是提高那些具有高期望学习进展（通过绝对 TD 误差的代理衡量）的经验元组的回放概率。与均匀经验回放相比，这在 Atari 基准测试套件的大多数游戏中带来了更快的学习速度和更好的最终策略质量。

为加强我们的对决架构与算法创新互补的论点，我们展示了它在均匀回放和优先回放基线上都能提升性能（我们选择了更易实现的基于排名的变体），其中优先对决变体取得了新的最先进成果。

---

## 3. 对决网络架构

我们新架构背后的关键洞察如图 2 所示：对于许多状态，没有必要估计每个动作选择的价值。例如，在 Enduro 游戏设定中，只有在碰撞迫在眉睫时才需要知道是向左还是向右移动。在某些状态中，知道采取哪个动作至关重要，但在许多其他状态中，动作的选择不会产生任何影响。然而，对于基于自举（Bootstrapping）的算法，在每个状态中估计状态价值都是非常重要的。

为将这一洞察付诸实践，我们设计了一个如图 1 所示的单一 Q 网络架构，称之为对决网络。对决网络的底层是卷积层，与原始 DQN (Mnih et al., 2015) 相同。然而，我们没有在卷积层之后接一个单序列的全连接层，而是使用两个序列（即两个流）的全连接层。这两个流的构造使其能够分别提供价值函数和优势函数的独立估计。最终，两个流合并以产生单一的 Q 函数输出。如 Mnih et al. (2015) 中一样，网络的输出是一组 Q 值，每个动作对应一个。

由于对决网络的输出是 Q 函数，它可以使用许多现有算法进行训练，如 DDQN 和 SARSA。此外，它可以利用这些算法的任何改进，包括更好的回放记忆、更好的探索策略、内在动机等。

将两个全连接层流合并以输出 Q 估计的模块需要非常精心的设计。

从优势的表达式 $Q^\pi(s, a) = V^\pi(s) + A^\pi(s, a)$ 和状态价值 $V^\pi(s) = \mathbb{E}_{a \sim \pi(s)}[Q^\pi(s, a)]$，可得 $\mathbb{E}_{a \sim \pi(s)}[A^\pi(s, a)] = 0$。此外，对于确定性策略 $a^* = \arg\max_{a' \in \mathcal{A}} Q(s, a')$，有 $Q(s, a^*) = V(s)$，因此 $A(s, a^*) = 0$。

考虑图 1 所示的对决网络，其中一个全连接层流输出标量 $V(s; \theta, \beta)$，另一个流输出 $|\mathcal{A}|$ 维向量 $A(s, a; \theta, \alpha)$。这里 $\theta$ 表示卷积层的参数，$\alpha$ 和 $\beta$ 分别是两个全连接层流的参数。

利用优势的定义，我们可能倾向于按如下方式构造聚合模块：

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha)$$
$\hfill (7)$

注意该表达式适用于所有 $(s, a)$ 实例；即要以矩阵形式表达公式 (7)，需要将标量 $V(s; \theta, \beta)$ 复制 $|\mathcal{A}|$ 次。

然而，我们需要牢记 $Q(s, a; \theta, \alpha, \beta)$ 仅是真实 Q 函数的参数化估计。此外，得出 $V(s; \theta, \beta)$ 是状态价值函数的良好估计器，或者 $A(s, a; \theta, \alpha)$ 提供了优势函数的合理估计，这样的结论是错误的。

公式 (7) 在可辨识性（Identifiability）意义上是不可辨识的：给定 $Q$，我们无法唯一地恢复 $V$ 和 $A$。为证明这一点，可以向 $V(s; \theta, \beta)$ 加一个常数，再从 $A(s, a; \theta, \alpha)$ 减去相同的常数，该常数相互抵消，得到相同的 $Q$ 值。这种可辨识性的缺失反映在直接使用该方程时的糟糕实际性能上。

为解决可辨识性问题，我们可以强制优势函数估计器在被选动作处的优势为零。即，让网络的最后一个模块实现以下前向映射：

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left(A(s, a; \theta, \alpha) - \max_{a' \in |\mathcal{A}|} A(s, a'; \theta, \alpha)\right)$$
$\hfill (8)$

此时，对于 $a^* = \arg\max_{a' \in \mathcal{A}} Q(s, a'; \theta, \alpha, \beta) = \arg\max_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha)$，我们得到 $Q(s, a^*; \theta, \alpha, \beta) = V(s; \theta, \beta)$。因此，流 $V(s; \theta, \beta)$ 提供了价值函数的估计，而另一个流产生了优势函数的估计。

一个替代模块用均值（Average）替换了 max 算子：

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left(A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha)\right)$$
$\hfill (9)$

一方面，这丧失了 $V$ 和 $A$ 的原始语义，因为它们现在偏离目标一个常数；但另一方面，它提高了优化的稳定性：使用公式 (9)，优势只需要以均值的速度变化，而不必补偿公式 (8) 中最优动作优势的任何变化。我们还实验了公式 (8) 的 softmax 版本，但发现其结果与更简单的公式 (9) 模块相似。因此，本文报告的所有实验都使用公式 (9) 的模块。

值得注意的是，虽然公式 (9) 中减去均值有助于可辨识性，但它不改变 $A$（进而 $Q$）值的相对排序，从而保持了基于公式 (7) 的 Q 值的任何贪心或 $\epsilon$-贪心策略。在行动时，只需评估优势流即可做出决策。

重要的是，公式 (9) 被视为并实现为网络的一部分，而非单独的算法步骤。对决架构的训练与标准 Q 网络（如 Mnih et al., 2015 的深度 Q 网络）一样，只需要反向传播。估计 $V(s; \theta, \beta)$ 和 $A(s, a; \theta, \alpha)$ 是自动计算的，无需任何额外监督或算法修改。

由于对决架构与标准 Q 网络共享相同的输入输出接口，我们可以复用所有基于 Q 网络的学习算法（如 DDQN 和 SARSA）来训练对决架构。

---

## 4. 实验

我们现在展示对决网络的实际性能。我们从一个简单的策略评估任务开始，然后展示学习通用 Atari 游戏策略的大规模结果。

### 4.1 策略评估

我们首先在策略评估（Policy Evaluation）任务上衡量对决架构的性能。选择该特定任务是因为它对于评估网络架构非常有用，没有探索策略选择以及策略改进与策略评估之间交互等混杂因素。

在本实验中，我们使用时序差分学习（Temporal Difference Learning，不含资格迹，即 $\lambda = 0$）来学习 Q 值。更具体地说，给定行为策略 $\pi$，我们通过优化公式 (4) 的代价序列来估计状态-动作价值 $Q^\pi(\cdot, \cdot)$，其目标为：

$$y_i = r + \gamma \mathbb{E}_{a' \sim \pi(s')}[Q(s', a'; \theta_i)]$$

上述更新规则与期望 SARSA（Expected SARSA）(van Seijen et al., 2009) 相同，但我们不像期望 SARSA 那样修改行为策略。

为评估学习到的 Q 值，我们选择了一个可以单独计算所有 $(s, a) \in \mathcal{S} \times \mathcal{A}$ 的精确 $Q^\pi(s, a)$ 值的简单环境。该环境称为走廊（Corridor），由三个相连的走廊组成。智能体从环境的左下角出发，必须移动到右上角以获得最大奖励。共有 5 个可用动作：上、下、左、右和无操作（No-op）。我们还可以自由添加任意数量的无操作动作。在我们的设置中，两个垂直部分各有 10 个状态，水平部分有 50 个状态。

我们使用 $\epsilon$-贪心策略作为行为策略 $\pi$，以概率 $\epsilon$ 选择随机动作，或以概率 $1 - \epsilon$ 根据最优 Q 函数 $\arg\max_{a \in \mathcal{A}} Q^*(s, a)$ 选择动作。在我们的实验中，$\epsilon$ 设为 0.001。

我们在走廊环境的三个变体（分别有 5、10 和 20 个动作）上比较了单流 Q 架构和对决架构。10 和 20 个动作的变体是通过向原始环境添加无操作动作形成的。我们使用与真实状态价值的平方误差（Squared Error, SE）来衡量性能：$\sum_{s \in \mathcal{S}, a \in \mathcal{A}}(Q(s, a; \theta) - Q^\pi(s, a))^2$。单流架构是一个三层 MLP，每个隐藏层 50 个单元。对决架构也由三层组成，但在第一个 50 单元的隐藏层之后，网络分支为两个流，每个流是一个两层 MLP，各有 25 个隐藏单元。比较结果汇总在图 3 中。

> **图 3 描述**：(a) 走廊环境示意图，星号标记起始状态，状态的红色程度表示智能体到达时获得的奖励，到达任一奖励状态后游戏终止。(b)(c)(d) 分别为 5、10 和 20 个动作的策略评估平方误差（对数-对数坐标）。对决网络 (Duel) 持续优于传统单流网络 (Single)，且性能差距随动作数量增加而增大。

结果表明，在 5 个动作时，两种架构的收敛速度大致相同。然而，当增加动作数量时，对决架构的表现优于传统 Q 网络。在对决网络中，流 $V(s; \theta, \beta)$ 学习到一个在状态 $s$ 处跨许多相似动作共享的通用价值，从而导致更快的收敛。这是一个非常有前景的结果，因为许多具有大动作空间的控制任务具有这种特性，因此我们应该期望对决网络通常比传统单流网络收敛更快。在下一节中，我们将看到对决网络在大量 Atari 游戏中取得了实质性的性能提升。

### 4.2 通用 Atari 游戏

我们在街机学习环境（Arcade Learning Environment, ALE）(Bellemare et al., 2013) 上对所提出的方法进行了全面评估，该环境包含 57 个 Atari 游戏。挑战在于部署一个具有固定超参数集的单一算法和架构，仅通过原始像素观测和游戏奖励来学会玩所有游戏。这个环境要求非常高，因为它既包含大量高度多样化的游戏，观测也是高维的。

我们紧密遵循 van Hasselt et al. (2015) 的设置并与其使用单流 Q 网络的结果进行比较。我们使用附录 A 中的 DDQN 算法训练对决网络。在本节末尾，我们还纳入了优先经验回放 (Schaul et al., 2016)。

我们的网络架构具有与 DQN (Mnih et al., 2015; van Hasselt et al., 2015) 相同的底层卷积结构。共有 3 个卷积层后接 2 个全连接层。第一个卷积层有 32 个 $8 \times 8$ 滤波器，步幅为 4；第二个有 64 个 $4 \times 4$ 滤波器，步幅为 2；第三个也是最后一个卷积层有 64 个 $3 \times 3$ 滤波器，步幅为 1。如图 1 所示，对决网络分为两个全连接层流。价值流和优势流各有一个 512 单元的全连接层。价值流和优势流的最终隐藏层都是全连接的，价值流有一个输出，优势流有与有效动作数量相同的输出（ALE 环境中动作数量在 3-18 之间）。我们使用公式 (9) 描述的模块合并价值流和优势流。在所有相邻层之间插入修正线性单元（Rectifier）非线性激活函数 (Fukushima, 1980)。

我们采用 van Hasselt et al. (2015) 的优化器和超参数，但学习率略低（对双重 DQN 不做此调整，因为这可能降低其性能）。由于价值流和优势流在反向传播中都将梯度传播到最后一个卷积层，我们将进入最后一个卷积层的合并梯度缩放 $1/\sqrt{2}$。这个简单的启发式方法略微提高了稳定性。此外，我们将梯度裁剪（Gradient Clipping）使其范数不超过 10。这种裁剪在深度 RL 中不是标准做法，但在循环网络训练中很常见 (Bengio et al., 2013)。

为隔离对决架构的贡献，我们使用与上述完全相同的程序重新训练了一个单流网络的 DDQN。具体来说，我们应用梯度裁剪，并为网络的第一个全连接层使用 1024 个隐藏单元，使两种架构（对决和单流）具有大致相同数量的参数。我们将这个重新训练的模型称为 Single Clip，而 van Hasselt et al. (2015) 的原始训练模型称为 Single。

如 van Hasselt et al. (2015) 一样，我们以最多 30 个无操作动作开始游戏，为智能体提供随机起始位置。为评估我们的方法，我们测量相对于人类和基线智能体分数中较高者的分数改进百分比（正或负）：

$$\frac{\text{Score}_{\text{Agent}} - \text{Score}_{\text{Baseline}}}{\max\{\text{Score}_{\text{Human}}, \text{Score}_{\text{Baseline}}\} - \text{Score}_{\text{Random}}}$$
$\hfill (10)$

取人类和基线智能体分数的最大值，可以防止当被评估智能体和基线都表现不佳时，微小变化显示为巨大改进。例如，一个达到 2% 人类水平的智能体不应被解释为基线达到 1% 人类水平时的两倍。我们也选择不单独以人类水平百分比来衡量性能，因为在某些游戏上相对于基线的微小差异可能转化为数百个百分点的人类性能差异。

57 个游戏的完整结果汇总在表 1 中，详细结果见附录。使用 30 个无操作性能指标，对决网络 (Duel Clip) 明显优于同等容量的 Single Clip 网络，也显著优于 van Hasselt et al. (2015) 的基线 (Single)。作为对比，我们还展示了 Mnih et al. (2015) 的深度 Q 网络结果，称为 Nature DQN。

> **图 4 描述**：使用公式 (10) 所述指标，对决架构相对于 van Hasselt et al. (2015) 的基线 Single 网络的改进。右侧的条形图表示对决网络超越单流网络的幅度。

**表 1. 所有 57 个 Atari 游戏的平均和中位数分数，以人类水平百分比衡量。**

| | 30 no-ops Mean | 30 no-ops Median | Human Starts Mean | Human Starts Median |
|---|---|---|---|---|
| Prior. Duel Clip | 591.9% | 172.1% | 567.0% | 115.3% |
| Prior. Single | 434.6% | 123.7% | 386.7% | 112.9% |
| Duel Clip | 373.1% | 151.5% | 343.8% | 117.1% |
| Single Clip | 341.2% | 132.6% | 302.8% | 114.1% |
| Single | 307.3% | 117.8% | 332.9% | 110.9% |
| Nature DQN | 227.9% | 79.1% | 219.6% | 68.5% |

Duel Clip 在 75.4% 的游戏（57 个中的 43 个）上优于 Single Clip，在 80.7% 的游戏（57 个中的 46 个）上取得了比 Single 基线更高的分数。在所有 18 个动作的游戏中，Duel Clip 在 86.6% 的时间（30 个中的 26 个）表现更好。这与前一节的发现一致。总体而言，我们的智能体 (Duel Clip) 在 57 个游戏中的 42 个上达到了人类水平。所有游戏的原始分数以及人类水平百分比的测量结果见附录。

**对人类起始点的鲁棒性。** 30 个无操作指标的一个缺点是智能体不一定需要很好地泛化才能玩 Atari 游戏。由于 Atari 环境的确定性性质，从唯一的起始点出发，智能体可以通过简单地记忆动作序列来获得好的表现。

为获得更鲁棒的测量，我们采用 Nair et al. (2015) 的方法。具体来说，对于每个游戏，我们使用从人类专家轨迹中采样的 100 个起始点。从每个起始点开始，评估回合最多运行 108,000 帧。智能体仅根据起始点之后获得的奖励进行评估。我们将此指标称为 Human Starts。

如表 1 所示，在 Human Starts 指标下，Duel Clip 再次优于单流变体。特别是，我们的智能体在 70.2% 的游戏（57 个中的 40 个）上优于 Single 基线，在 18 个动作的游戏上，Duel Clip 在 83.3%（30 个中的 25 个）上表现更好。

**与优先经验回放结合。** 对决架构可以很容易地与其他算法改进结合。特别是，经验回放的优先化已被证明能显著提高 Atari 游戏的性能 (Schaul et al., 2016)。此外，由于优先化和对决架构解决的是学习过程的不同方面，二者的结合很有前景。因此，在最后一个实验中，我们研究了对决架构与优先经验回放的集成。我们使用 DDQN 的优先变体 (Prior. Single) 作为新的基线算法，将经验元组的均匀采样替换为基于排名的优先采样。我们保持 Schaul et al. (2016) 中描述的优先回放的所有参数，即优先级指数为 0.7，重要性采样指数从 0.5 退火到 1。我们将此基线与对决架构结合，并再次使用梯度裁剪 (Prior. Duel Clip)。

值得注意的是，尽管这些扩展（优先化、对决和梯度裁剪）在目标上是正交的，但它们之间存在微妙的交互。例如，优先化与梯度裁剪存在交互，因为更频繁地采样具有高绝对 TD 误差的转移会导致具有更高范数的梯度。为避免不利交互，我们在 9 个游戏的子集上粗略地重新调整了学习率和梯度裁剪范数。经粗调后，我们确定学习率为 $6.25 \times 10^{-5}$，梯度裁剪范数为 10（与前一节相同）。

在所有 57 个 Atari 游戏上评估时，我们的优先对决智能体的表现明显优于优先基线智能体和单独的对决智能体。完整的平均和中位数人类水平百分比性能见表 1。当使用最多 30 个无操作动作初始化游戏时，我们观察到平均和中位数分数分别为 591% 和 172%。优先基线和优先对决版本之间的直接比较（使用公式 10 描述的指标）见图 5。

优先回放与对决网络的结合在流行的 ALE 基准测试上取得了相比先前最先进水平的巨大改进。

> **图 5 描述**：使用与图 4 相同的指标，对决架构相对于优先 DDQN 基线的改进。对决架构在大多数游戏上相对于单流基线带来了显著改进。

**显著性图。** 为更好地理解价值流和优势流的角色，我们计算了显著性图 (Simonyan et al., 2013)。具体来说，为可视化价值流所看到的图像显著部分，我们计算 $\hat{V}$ 相对于输入帧的雅可比矩阵的绝对值：$|\nabla_s \hat{V}(s; \theta)|$。类似地，为可视化优势流所看到的图像显著部分，我们计算 $|\nabla_s \hat{A}(s, \arg\max_{a'} \hat{A}(s, a'); \theta)|$。这两个量与输入帧具有相同的维度，因此可以很容易地与输入帧一起可视化。

我们将灰度输入帧放在绿色和蓝色通道中，将显著性图放在红色通道中，三个通道共同形成 RGB 图像。图 2 描绘了 Enduro 游戏在两个不同时间步的价值和优势显著性图。如引言中所观察到的，价值流关注地平线（那里出现的汽车可能影响未来表现），价值流也关注分数。另一方面，优势流更关心处于直接碰撞路线上的汽车。

---

## 5. 讨论

对决架构的优势部分在于其高效学习状态价值函数的能力。在对决架构中，每次 Q 值更新时，价值流 $V$ 都会被更新——这与单流架构形成对比，在单流架构中只有一个动作的值被更新，所有其他动作的值保持不变。在我们的方法中，价值流更频繁地更新，从而为 $V$ 分配了更多资源，进而允许更好地近似状态价值，而这对于基于时序差分的方法（如 Q 学习）的正常工作是必需的 (Sutton & Barto, 1998)。这一现象在实验中得到了反映，其中对决架构相对于单流 Q 网络的优势随着动作数量的增加而增长。

此外，给定状态的 Q 值之间的差异通常相对于 Q 的幅度非常小。例如，在 Seaquest 游戏上使用 DDQN 训练后，跨访问状态的平均动作间隙（给定状态下最佳和第二最佳动作的 Q 值差距）约为 0.04，而这些状态的平均状态价值约为 15。这种尺度差异意味着更新中少量噪声可能导致动作的重新排序，从而使近贪心策略突然切换。对决架构及其独立的优势流对此类效应具有鲁棒性。

---

## 6. 结论

我们引入了一种新的神经网络架构，在深度 Q 网络中解耦价值和优势，同时共享一个公共的特征学习模块。新的对决架构结合一些算法改进，在极具挑战性的 Atari 域上带来了相对于现有深度 RL 方法的巨大改进。本文呈现的结果是该热门测试域上新的最先进水平。

---

## 参考文献

- Ba, J., Mnih, V., and Kavukcuoglu, K. Multiple object recognition with visual attention. In ICLR, 2015.
- Baird, L.C. Advantage updating. Technical Report WL-TR-93-1146, Wright-Patterson Air Force Base, 1993.
- Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253–279, 2013.
- Bellemare, M. G., Ostrovski, G., Guez, A., Thomas, P. S., and Munos, R. Increasing the action gap: New operators for reinforcement learning. In AAAI, 2016. To appear.
- Bengio, Y., Boulanger-Lewandowski, N., and Pascanu, R. Advances in optimizing recurrent networks. In ICASSP, pp. 8624–8628, 2013.
- Fukushima, K. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36:193–202, 1980.
- Guo, X., Singh, S., Lee, H., Lewis, R. L., and Wang, X. Deep learning for real-time Atari game play using offline Monte-Carlo tree search planning. In NIPS, pp. 3338–3346. 2014.
- Harmon, M.E. and Baird, L.C. Multi-player residual advantage learning with general function approximation. Technical Report WL-TR-1065, Wright-Patterson Air Force Base, 1996.
- Harmon, M.E., Baird, L.C., and Klopf, A.H. Advantage updating applied to a differential game. In G. Tesauro, D.S. Touretzky and Leen, T.K. (eds.), NIPS, 1995.
- LeCun, Y., Bengio, Y., and Hinton, G. Deep learning. Nature, 521(7553):436–444, 2015.
- Levine, S., Finn, C., Darrell, T., and Abbeel, P. End-to-end training of deep visuomotor policies. arXiv preprint arXiv:1504.00702, 2015.
- Lin, L.J. Reinforcement learning for robots using neural networks. PhD thesis, School of Computer Science, Carnegie Mellon University, 1993.
- Maddison, C. J., Huang, A., Sutskever, I., and Silver, D. Move Evaluation in Go Using Deep Convolutional Neural Networks. In ICLR, 2015.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., and Hassabis, D. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.
- Nair, A., Srinivasan, P., Blackwell, S., Alcicek, C., Fearon, R., Maria, A. De, Panneershelvam, V., Suleyman, M., Beattie, C., Petersen, S., Legg, S., Mnih, V., Kavukcuoglu, K., and Silver, D. Massively parallel methods for deep reinforcement learning. In Deep Learning Workshop, ICML, 2015.
- Schaul, T., Quan, J., Antonoglou, I., and Silver, D. Prioritized experience replay. In ICLR, 2016.
- Schulman, J., Moritz, P., Levine, S., Jordan, M. I., and Abbeel, P. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
- Silver, D., Huang, A., Maddison, C.J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484–489, 01 2016.
- Simonyan, K., Vedaldi, A., and Zisserman, A. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.
- Stadie, B. C., Levine, S., and Abbeel, P. Incentivizing exploration in reinforcement learning with deep predictive models. arXiv preprint arXiv:1507.00814, 2015.
- Sutton, R. S. and Barto, A. G. Introduction to reinforcement learning. MIT Press, 1998.
- Sutton, R. S., Mcallester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. In NIPS, pp. 1057–1063, 2000.
- van Hasselt, H. Double Q-learning. NIPS, 23:2613–2621, 2010.
- van Hasselt, H., Guez, A., and Silver, D. Deep reinforcement learning with double Q-learning. arXiv preprint arXiv:1509.06461, 2015.
- van Seijen, H., van Hasselt, H., Whiteson, S., and Wiering, M. A theoretical and empirical analysis of Expected Sarsa. In IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning, pp. 177–184. 2009.
- Watter, M., Springenberg, J. T., Boedecker, J., and Riedmiller, M. A. Embed to control: A locally linear latent dynamics model for control from raw images. In NIPS, 2015.

---

## 附录 A. 双重 DQN 算法

**算法 1：双重 DQN 算法**

**输入**：$D$ — 空回放缓冲区；$\theta$ — 初始网络参数，$\theta^-$ — $\theta$ 的副本

**输入**：$N_r$ — 回放缓冲区最大容量；$N_b$ — 训练批量大小；$N^-$ — 目标网络替换频率

**for** 回合 $e \in \{1, 2, \ldots, M\}$ **do**

$\quad$ 初始化帧序列 $x \leftarrow ()$

$\quad$ **for** $t \in \{0, 1, \ldots\}$ **do**

$\quad\quad$ 设置状态 $s \leftarrow x$，采样动作 $a \sim \pi_B$

$\quad\quad$ 从环境 $\mathcal{E}$ 中给定 $(s, a)$ 采样下一帧 $x_t$ 并接收奖励 $r$，将 $x_t$ 追加到 $x$

$\quad\quad$ **if** $|x| > N_f$ **then** 删除 $x$ 中最旧的帧 $x_{t_{\min}}$ **end**

$\quad\quad$ 设置 $s' \leftarrow x$，将转移元组 $(s, a, r, s')$ 添加到 $D$，若 $|D| \geq N_r$ 则替换最旧的元组

$\quad\quad$ 从 $D$ 中均匀随机采样 $N_b$ 个元组 $(s, a, r, s') \sim \text{Unif}(D)$

$\quad\quad$ 为 $N_b$ 个元组中的每一个构造目标值：

$\quad\quad$ 定义 $a_{\max}(s'; \theta) = \arg\max_{a'} Q(s', a'; \theta)$

$$y_j = \begin{cases} r & \text{if } s' \text{ is terminal} \\ r + \gamma Q(s', a_{\max}(s'; \theta); \theta^-) & \text{otherwise} \end{cases}$$

$\quad\quad$ 使用损失 $\|y_j - Q(s, a; \theta)\|^2$ 执行梯度下降步骤

$\quad\quad$ 每 $N^-$ 步替换目标参数 $\theta^- \leftarrow \theta$

$\quad$ **end**

**end**

---

## 附录：实验结果详表

### 表 2. 所有游戏的原始分数（30 no-op 起始）

| 游戏 | 动作数 | Random | Human | DQN | DDQN | Duel | Prior. | Prior. Duel. |
|------|--------|--------|-------|-----|------|------|--------|-------------|
| Alien | 18 | 227.8 | 7,127.7 | 1,620.0 | 3,747.7 | 4,461.4 | 4,203.8 | 3,941.0 |
| Amidar | 10 | 5.8 | 1,719.5 | 978.0 | 1,793.3 | 2,354.5 | 1,838.9 | 2,296.8 |
| Assault | 7 | 222.4 | 742.0 | 4,280.4 | 5,393.2 | 4,621.0 | 7,672.1 | 11,477.0 |
| Asterix | 9 | 210.0 | 8,503.3 | 4,359.0 | 17,356.5 | 28,188.0 | 31,527.0 | 375,080.0 |
| Asteroids | 14 | 719.1 | 47,388.7 | 1,364.5 | 734.7 | 2,837.7 | 2,654.3 | 1,192.7 |
| Atlantis | 4 | 12,850.0 | 29,028.1 | 279,987.0 | 106,056.0 | 382,572.0 | 357,324.0 | 395,762.0 |
| Bank Heist | 18 | 14.2 | 753.1 | 455.0 | 1,030.6 | 1,611.9 | 1,054.6 | 1,503.1 |
| Battle Zone | 18 | 2,360.0 | 37,187.5 | 29,900.0 | 31,700.0 | 37,150.0 | 31,530.0 | 35,520.0 |
| Beam Rider | 9 | 363.9 | 16,926.5 | 8,627.5 | 13,772.8 | 12,164.0 | 23,384.2 | 30,276.5 |
| Berzerk | 18 | 123.7 | 2,630.4 | 585.6 | 1,225.4 | 1,472.6 | 1,305.6 | 3,409.0 |
| Bowling | 6 | 23.1 | 160.7 | 50.4 | 68.1 | 65.5 | 47.9 | 46.7 |
| Boxing | 18 | 0.1 | 12.1 | 88.0 | 91.6 | 99.4 | 95.6 | 98.9 |
| Breakout | 4 | 1.7 | 30.5 | 385.5 | 418.5 | 345.3 | 373.9 | 366.0 |
| Centipede | 18 | 2,090.9 | 12,017.0 | 4,657.7 | 5,409.4 | 7,561.4 | 4,463.2 | 7,687.5 |
| Chopper Command | 18 | 811.0 | 7,387.8 | 6,126.0 | 5,809.0 | 11,215.0 | 8,600.0 | 13,185.0 |
| Crazy Climber | 9 | 10,780.5 | 35,829.4 | 110,763.0 | 117,282.0 | 143,570.0 | 141,161.0 | 162,224.0 |
| Defender | 18 | 2,874.5 | 18,688.9 | 23,633.0 | 35,338.5 | 42,214.0 | 31,286.5 | 41,324.5 |
| Demon Attack | 6 | 152.1 | 1,971.0 | 12,149.4 | 58,044.2 | 60,813.3 | 71,846.4 | 72,878.6 |
| Double Dunk | 18 | -18.6 | -16.4 | -6.6 | -5.5 | 0.1 | 18.5 | -12.5 |
| Enduro | 9 | 0.0 | 860.5 | 729.0 | 1,211.8 | 2,258.2 | 2,093.0 | 2,306.4 |
| Fishing Derby | 18 | -91.7 | -38.7 | -4.9 | 15.5 | 46.4 | 39.5 | 41.3 |
| Freeway | 3 | 0.0 | 29.6 | 30.8 | 33.3 | 0.0 | 33.7 | 33.0 |
| Frostbite | 18 | 65.2 | 4,334.7 | 797.4 | 1,683.3 | 4,672.8 | 4,380.1 | 7,413.0 |
| Gopher | 8 | 257.6 | 2,412.5 | 8,777.4 | 14,840.8 | 15,718.4 | 32,487.2 | 104,368.2 |
| Gravitar | 18 | 173.0 | 3,351.4 | 473.0 | 412.0 | 588.0 | 548.5 | 238.0 |
| H.E.R.O. | 18 | 1,027.0 | 30,826.4 | 20,437.8 | 20,130.2 | 20,818.2 | 23,037.7 | 21,036.5 |
| Ice Hockey | 18 | -11.2 | 0.9 | -1.9 | -2.7 | 0.5 | 1.3 | -0.4 |
| James Bond | 18 | 29.0 | 302.8 | 768.5 | 1,358.0 | 1,312.5 | 5,148.0 | 812.0 |
| Kangaroo | 18 | 52.0 | 3,035.0 | 7,259.0 | 12,992.0 | 14,854.0 | 16,200.0 | 1,792.0 |
| Krull | 18 | 1,598.0 | 2,665.5 | 8,422.3 | 7,920.5 | 11,451.9 | 9,728.0 | 10,374.4 |
| Kung-Fu Master | 14 | 258.5 | 22,736.3 | 26,059.0 | 29,710.0 | 34,294.0 | 39,581.0 | 48,375.0 |
| Montezuma's Revenge | 18 | 0.0 | 4,753.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ms. Pac-Man | 9 | 307.3 | 6,951.6 | 3,085.6 | 2,711.4 | 6,283.5 | 6,518.7 | 3,327.3 |
| Name This Game | 6 | 2,292.3 | 8,049.0 | 8,207.8 | 10,616.0 | 11,971.1 | 12,270.5 | 15,572.5 |
| Phoenix | 8 | 761.4 | 7,242.6 | 8,485.2 | 12,252.5 | 23,092.2 | 18,992.7 | 70,324.3 |
| Pitfall! | 18 | -229.4 | 6,463.7 | -286.1 | -29.9 | 0.0 | -356.5 | 0.0 |
| Pong | 3 | -20.7 | 14.6 | 19.5 | 20.9 | 21.0 | 20.6 | 20.9 |
| Private Eye | 18 | 24.9 | 69,571.3 | 146.7 | 129.7 | 103.0 | 200.0 | 206.0 |
| Q*Bert | 6 | 163.9 | 13,455.0 | 13,117.3 | 15,088.5 | 19,220.3 | 16,256.5 | 18,760.3 |
| River Raid | 18 | 1,338.5 | 17,118.0 | 7,377.6 | 14,884.5 | 21,162.6 | 14,522.3 | 20,607.6 |
| Road Runner | 18 | 11.5 | 7,845.0 | 39,544.0 | 44,127.0 | 69,524.0 | 57,608.0 | 62,151.0 |
| Robotank | 18 | 2.2 | 11.9 | 63.9 | 65.1 | 65.3 | 62.6 | 27.5 |
| Seaquest | 18 | 68.4 | 42,054.7 | 5,860.6 | 16,452.7 | 50,254.2 | 26,357.8 | 931.6 |
| Skiing | 3 | -17,098.1 | -4,336.9 | -13,062.3 | -9,021.8 | -8,857.4 | -9,996.9 | -19,949.9 |
| Solaris | 18 | 1,236.3 | 12,326.7 | 3,482.8 | 3,067.8 | 2,250.8 | 4,309.0 | 133.4 |
| Space Invaders | 6 | 148.0 | 1,668.7 | 1,692.3 | 2,525.5 | 6,427.3 | 2,865.8 | 15,311.5 |
| Star Gunner | 18 | 664.0 | 10,250.0 | 54,282.0 | 60,142.0 | 89,238.0 | 63,302.0 | 125,117.0 |
| Surround | 5 | -10.0 | 6.5 | -5.6 | -2.9 | 4.4 | 8.9 | 1.2 |
| Tennis | 18 | -23.8 | -8.3 | 12.2 | -22.8 | 5.1 | 0.0 | 0.0 |
| Time Pilot | 10 | 3,568.0 | 5,229.2 | 4,870.0 | 8,339.0 | 11,666.0 | 9,197.0 | 7,553.0 |
| Tutankham | 8 | 11.4 | 167.6 | 68.1 | 218.4 | 211.4 | 204.6 | 245.9 |
| Up and Down | 6 | 533.4 | 11,693.2 | 9,989.9 | 22,972.2 | 44,939.6 | 16,154.1 | 33,879.1 |
| Venture | 18 | 0.0 | 1,187.5 | 163.0 | 98.0 | 497.0 | 54.0 | 48.0 |
| Video Pinball | 9 | 16,256.9 | 17,667.9 | 196,760.4 | 309,941.9 | 98,209.5 | 282,007.3 | 479,197.0 |
| Wizard Of Wor | 10 | 563.5 | 4,756.5 | 2,704.0 | 7,492.0 | 7,855.0 | 4,802.0 | 12,352.0 |
| Yars' Revenge | 18 | 3,092.9 | 54,576.9 | 18,098.9 | 11,712.6 | 49,622.1 | 11,357.0 | 69,618.1 |
| Zaxxon | 18 | 32.5 | 9,173.3 | 5,363.0 | 10,163.0 | 12,944.0 | 10,469.0 | 13,886.0 |

> 表 3（Human Starts 原始分数）、表 4（30 no-ops 归一化分数）、表 5（Human Starts 归一化分数）的完整数据可参见原文附录。
