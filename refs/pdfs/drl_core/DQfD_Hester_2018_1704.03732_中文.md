# 从演示中进行深度Q学习

**arXiv:1704.03732v4 [cs.AI] 2017年11月22日**

**作者：** Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John Agapiou, Joel Z. Leibo, Audrunas Gruslys

**机构：** Google DeepMind

---

## 摘要

深度强化学习（Deep Reinforcement Learning, Deep RL）在困难的决策问题上取得了多项引人注目的成功。然而，这些算法通常需要大量数据才能达到合理的性能。事实上，它们在学习过程中的表现可能极其糟糕。这在模拟器中或许是可以接受的，但严重限制了深度强化学习在许多现实任务中的适用性，因为在现实任务中智能体必须在真实环境中学习。本文研究了一种设定：智能体可以访问系统先前控制器的数据。我们提出了一种名为**从演示中进行深度Q学习（Deep Q-learning from Demonstrations, DQfD）**的算法，它利用少量演示数据来大幅加速学习过程，即使演示数据量相对较少也能奏效，并且能够借助优先经验回放（Prioritized Replay）机制在学习过程中自动评估演示数据的必要比例。DQfD通过将时序差分（Temporal Difference, TD）更新与演示者动作的监督分类相结合来工作。我们表明，DQfD的初始性能优于优先对偶双重深度Q网络（Prioritized Dueling Double Deep Q-Networks, PDD DQN），在42个游戏中的41个游戏上，DQfD在前一百万步的得分更高，平均而言PDD DQN需要8300万步才能追上DQfD的性能。DQfD在42个游戏中的14个游戏中学会了超越给定的最佳演示。此外，DQfD利用人类演示在11个游戏上实现了当时最先进的结果。最后，我们表明DQfD优于其他三种将演示数据融入DQN的相关算法。

---

## 1 引言

在过去几年中，在序列决策问题和控制的策略学习方面取得了许多成功。著名的例子包括：用于通用Atari游戏的深度无模型Q学习（Mnih et al. 2015），用于机器人电机控制的端到端策略搜索（Levine et al. 2016），基于嵌入的模型预测控制（Watter et al. 2015），以及结合搜索的策略方法在围棋比赛中击败了顶级人类专家（Silver et al. 2016）。这些方法成功的一个重要因素是利用了深度学习在可扩展性和性能方面的最新进展（LeCun, Bengio, and Hinton 2015）。Mnih et al. (2015) 的方法利用先前经验构建数据集，使用批量强化学习以监督方式从这些数据中训练大型卷积神经网络（Convolutional Neural Network, CNN）。通过从该数据集而非当前经验中采样，缓解了状态分布偏差导致的值相关性问题，从而学到了良好的（在许多情况下超越人类的）控制策略。

然而，将这些算法应用于数据中心、自动驾驶车辆（Hester and Stone 2013）、直升机（Abbeel et al. 2007）或推荐系统（Shani, Heckerman, and Brafman 2005）等现实场景仍然困难。通常这些算法需要在模拟中经历数百万步非常差的性能后才能学到好的控制策略。当存在完全精确的模拟器时这种情况是可以接受的；然而，许多现实问题并不具备这样的模拟器。相反，在这些情况下，智能体必须在真实环境中学习，其动作会产生真实的后果，这要求智能体从学习开始就具备良好的在线性能。虽然精确的模拟器难以获得，但这些问题中大多数都有系统在先前控制器（人类或机器）控制下运行的数据，且这些控制器表现得相当不错。

在本工作中，我们利用这些演示数据来预训练智能体，使其从学习开始就能在任务中表现良好，然后继续从自生成的数据中改进。在这一框架下实现学习，为将强化学习应用于许多演示数据常见但精确模拟器不存在的现实问题开辟了可能性。

我们提出了一种新的深度强化学习算法——从演示中进行深度Q学习（DQfD），它即使利用非常少量的演示数据也能大幅加速学习。DQfD首先仅在演示数据上进行预训练，使用时序差分（TD）损失和监督损失的组合。监督损失使算法学会模仿演示者，而TD损失使其学到一个自洽的值函数，从而在智能体开始与环境交互后可以继续通过强化学习改进。预训练完成后，智能体开始用学到的策略与环境交互。智能体使用演示数据和自生成数据的混合来更新其网络。在实践中，选择学习过程中演示数据与自生成数据的比例对算法性能至关重要。我们的一个贡献是使用优先经验回放机制（Schaul et al. 2016）来自动控制这一比例。DQfD在42个游戏中的41个游戏的前一百万步中优于使用PDD DQN（Schaul et al. 2016; van Hasselt, Guez, and Silver 2016; Wang et al. 2016）的纯强化学习，平均需要8300万步PDD DQN才能追上DQfD。此外，DQfD在42个游戏中的39个游戏的平均得分上优于纯模仿学习（Imitation Learning），并在42个游戏中的14个游戏中超越了给定的最佳演示。DQfD利用人类演示在42个游戏中的11个游戏上学到了当时最先进的策略。最后，我们表明DQfD优于其他三种将演示数据融入DQN的相关算法。

---

## 2 背景

本工作采用标准的马尔可夫决策过程（Markov Decision Process, MDP）形式化框架（Sutton and Barto 1998）。MDP由元组 $\langle S, A, R, T, \gamma \rangle$ 定义，包括状态集合 $S$、动作集合 $A$、奖励函数 $R(s, a)$、转移函数 $T(s, a, s') = P(s'|s, a)$ 以及折扣因子 $\gamma$。在每个状态 $s \in S$ 中，智能体执行一个动作 $a \in A$。执行该动作后，智能体获得奖励 $R(s, a)$ 并到达由概率分布 $P(s'|s, a)$ 决定的新状态 $s'$。策略 $\pi$ 为每个状态指定智能体将采取的动作。智能体的目标是找到将状态映射到动作的策略 $\pi$，以最大化智能体生命期内的期望折扣总奖励。给定状态-动作对 $(s, a)$ 的值 $Q^\pi(s, a)$ 是从 $(s, a)$ 出发按策略 $\pi$ 行动时期望未来奖励的估计。最优值函数 $Q^*(s, a)$ 在所有状态中提供最大值，通过求解贝尔曼方程（Bellman Equation）确定：

$$Q^*(s, a) = \mathbb{E}\left[R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q^*(s', a')\right]$$

最优策略为 $\pi(s) = \arg\max_{a \in A} Q^*(s, a)$。

DQN（Mnih et al. 2015）用深度神经网络近似值函数 $Q(s, a)$，该网络对给定状态输入 $s$ 输出一组动作值 $Q(s, \cdot; \theta)$，其中 $\theta$ 为网络参数。DQN有两个关键组成部分。首先，它使用一个单独的目标网络（Target Network），每 $\tau$ 步从常规网络复制，使目标Q值更稳定。其次，智能体将所有经验添加到经验回放缓冲区（Replay Buffer）$D_{replay}$ 中，然后均匀采样以对网络进行更新。

双重Q学习（Double Q-learning）更新（van Hasselt, Guez, and Silver 2016）使用当前网络计算下一状态值的argmax，并使用目标网络获取该动作的值。双重DQN损失为：

$$J_{DQ}(Q) = \left(R(s, a) + \gamma Q(s_{t+1}, a^{\max}_{t+1}; \theta') - Q(s, a; \theta)\right)^2$$

其中 $\theta'$ 为目标网络参数，$a^{\max}_{t+1} = \arg\max_a Q(s_{t+1}, a; \theta)$。将这两个变量使用的值函数分开可以减少常规Q学习更新产生的向上偏差。

优先经验回放（Prioritized Experience Replay）（Schaul et al. 2016）修改DQN智能体，使其更频繁地从回放缓冲区中采样更重要的转移。采样特定转移 $i$ 的概率与其优先级成正比：

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

其中优先级 $p_i = |\delta_i| + \epsilon$，$\delta_i$ 是该转移最后一次计算的TD误差，$\epsilon$ 是一个小正常数以确保所有转移都有一定的采样概率。为了补偿分布的变化，网络更新使用重要性采样（Importance Sampling）权重加权：

$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

其中 $N$ 为回放缓冲区大小，$\beta$ 控制重要性采样程度——$\beta = 0$ 时无重要性采样，$\beta = 1$ 时为完全重要性采样。$\beta$ 从 $\beta_0$ 线性退火到 $1$。

---

## 3 相关工作

**模仿学习（Imitation Learning）** 主要关注匹配演示者的性能。一种流行的算法DAGGER（Ross, Gordon, and Bagnell 2011）迭代地基于在专家原始状态空间之外查询专家策略来产生新策略，展示了这在在线学习意义上对验证数据可以达到无遗憾。DAGGER要求专家在训练期间可用以提供额外反馈。此外，它不将模仿与强化学习结合，意味着它永远无法像DQfD那样学会超越专家。

**Deeply AggreVaTeD**（Sun et al. 2017）将DAGGER扩展到深度神经网络和连续动作空间。它不仅像DAGGER一样需要一个始终可用的专家，而且专家还必须提供值函数以及动作。与DAGGER类似，Deeply AggreVaTeD仅进行模仿学习，无法学会超越专家。

另一种流行的范式是建立零和博弈（Zero-sum Game），其中学习者选择策略，对手选择奖励函数（Syed and Schapire 2007; Syed, Bowling, and Schapire 2008; Ho and Ermon 2016）。演示也被用于高维连续机器人控制问题的逆最优控制（Inverse Optimal Control）（Finn, Levine, and Abbeel 2016）。然而，这些方法仅进行模仿学习，不允许从任务奖励中学习。

最近，演示数据已被证明有助于强化学习中的困难探索问题（Subramanian, Jr., and Thomaz 2016）。在这一结合模仿与强化学习的问题上也引起了近期关注。例如，HAT算法直接从人类策略中迁移知识（Taylor, Suay, and Chernova 2011）。后续工作展示了专家建议或演示如何用于在强化学习问题中塑造奖励（Brys et al. 2015; Suay et al. 2016）。

另一种方法是塑造用于采样经验的策略（Cederborg et al. 2015），或使用从演示中进行策略迭代（Policy Iteration）（Kim et al. 2013; Chemali and Lezaric 2015）。

我们的算法适用于演示者使用的环境提供奖励的场景。这一框架在 Piot, Geist, and Pietquin (2014a) 中被恰当地称为**带专家演示的强化学习（Reinforcement Learning with Expert Demonstrations, RLED）**。我们的设置与 Piot, Geist, and Pietquin (2014a) 类似，都在无模型设置中将TD和分类损失结合在批量算法中；不同之处在于我们的智能体首先在演示数据上预训练，自生成数据的批量随时间增长，并作为经验回放来训练深度Q网络。此外，使用优先回放机制来平衡每个小批量中演示数据的数量。Piot, Geist, and Pietquin (2014b) 提出了有趣的结果，表明即使没有奖励，将TD损失添加到监督分类损失中也能改善模仿学习。

另一项与我们动机类似的工作是 Schaal (1996)。该工作关注机器人上的真实世界学习，因此也关心在线性能。与我们的工作类似，他们在让智能体与任务交互之前用演示数据预训练。然而，他们不使用监督学习来预训练算法，并且只在Cart-Pole上找到了一个预训练有帮助的案例。

在**单次模仿学习（One-shot Imitation Learning）**（Duan et al. 2017）中，除了当前状态外，智能体还接收整个演示作为输入。演示指定了所需的目标状态，但初始条件不同。该设置也使用演示，但需要具有不同初始条件和目标状态的任务分布，且智能体永远无法学会超越演示。

**AlphaGo**（Silver et al. 2016）采用了与我们类似的方法，在与真实任务交互之前从演示数据进行预训练。AlphaGo首先使用监督学习从3000万个专家动作的数据集中训练策略网络来预测专家采取的动作。然后将其作为起点，在自我博弈中应用策略梯度更新，并结合规划展开。在这里，我们没有可用于规划的模型，因此我们专注于无模型Q学习的情况。

**人类经验回放（Human Experience Replay, HER）**（Hosu and Rebedea 2016）是一种算法，智能体从混合了智能体和演示数据的回放缓冲区中采样，与我们的方法类似。其收益仅略好于随机智能体，且被他们的替代方法——人类检查点回放（Human Checkpoint Replay）——所超越，后者需要能够设置环境状态的能力。虽然他们的算法与我们类似地从两个数据集中采样，但它不预训练智能体或使用监督损失。我们的结果在更多种类的游戏上展示了更高的分数，且不需要完全访问环境。**回放缓冲区注入（Replay Buffer Spiking, RBS）**（Lipton et al. 2016）是另一种类似方法，其中DQN智能体的回放缓冲区用演示数据初始化，但他们不预训练智能体以获得良好的初始性能，也不永久保留演示数据。

与我们最密切相关的工作是一篇研讨会论文，提出了**带专家轨迹的加速DQN（Accelerated DQN with Expert Trajectories, ADET）**（Lakshminarayanan, Ozair, and Bengio 2016）。他们也在深度Q学习框架中结合了TD和分类损失。他们使用训练好的DQN智能体来生成演示数据，在大多数游戏中这优于人类数据。这也保证了演示者使用的策略可以被学徒智能体表示，因为它们都使用相同的状态输入和网络架构。他们使用交叉熵（Cross-entropy）分类损失而非DQfD使用的大间隔损失（Large Margin Loss），且不预训练智能体以在首次与环境交互时表现良好。

---

## 4 从演示中进行深度Q学习

在许多现实世界的强化学习场景中，我们可以获取系统由先前控制器操作的数据，但无法获得系统的精确模拟器。因此，我们希望智能体在真实系统上运行之前，尽可能多地从演示数据中学习。预训练阶段的目标是用满足贝尔曼方程的值函数来学习模仿演示者，以便在智能体开始与环境交互后可以用TD更新继续改进。

在预训练阶段，智能体从演示数据中采样小批量并通过应用四种损失来更新网络：

1. **1步双重Q学习损失（1-step Double Q-learning Loss）**
2. **n步双重Q学习损失（n-step Double Q-learning Loss）**
3. **监督大间隔分类损失（Supervised Large Margin Classification Loss）**
4. **L2正则化损失（L2 Regularization Loss）**

监督损失用于分类演示者的动作，而Q学习损失确保网络满足贝尔曼方程，可以作为TD学习的起点。

### 监督损失

监督损失对预训练产生效果至关重要。由于演示数据必然覆盖状态空间的狭窄部分且未采取所有可能的动作，许多状态-动作对从未被选取，没有数据将它们锚定到现实值。如果仅用朝向下一状态最大值的Q学习更新来预训练网络，网络将朝向这些未锚定变量中最高的值更新，并将这些值在Q函数中传播。我们添加了一个大间隔分类损失（Piot, Geist, and Pietquin 2014a）：

$$J_E(Q) = \max_{a \in A}[Q(s, a) + l(a_E, a)] - Q(s, a_E)$$

其中 $a_E$ 是专家演示者在状态 $s$ 中采取的动作，$l(a_E, a)$ 是间隔函数，当 $a = a_E$ 时为0，否则为正值。该损失迫使其他动作的值至少比演示者动作的值低一个间隔。添加这个损失将未见动作的值锚定到合理的值，并使值函数诱导的贪心策略模仿演示者。如果算法仅用这个监督损失进行预训练，则连续状态之间的值没有约束，Q网络将不满足贝尔曼方程，而这是通过TD学习在线改进策略所必需的。

### n步回报

添加n步回报（$n = 10$）有助于将专家轨迹的值传播到所有更早的状态，从而获得更好的预训练。n步回报为：

$$r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \max_a \gamma^n Q(s_{t+n}, a)$$

我们使用前向视图（Forward View）计算，类似于A3C（Mnih et al. 2016）。

### L2正则化

我们还添加了作用于网络权重和偏置的L2正则化损失，以帮助防止在相对较小的演示数据集上过拟合。

### 总体损失

用于更新网络的总体损失是所有四种损失的组合：

$$J(Q) = J_{DQ}(Q) + \lambda_1 J_n(Q) + \lambda_2 J_E(Q) + \lambda_3 J_{L2}(Q)$$

$\lambda$ 参数控制各损失之间的权重。我们在消融实验部分考察了移除部分损失的效果。

### 在线学习阶段

预训练阶段完成后，智能体开始在系统上行动，收集自生成数据，并将其添加到回放缓冲区 $D_{replay}$ 中。数据被添加到回放缓冲区直至满，然后智能体开始覆盖该缓冲区中的旧数据。然而，智能体永远不会覆盖演示数据。对于比例优先采样（Proportional Prioritized Sampling），不同的小正常数 $\epsilon_a$ 和 $\epsilon_d$ 被添加到智能体和演示转移的优先级中，以控制演示数据与智能体数据的相对采样。所有损失都应用于两个阶段的演示数据，而监督损失不应用于自生成数据（$\lambda_2 = 0$）。

### DQfD与PDD DQN的六个关键区别

1. **演示数据**：DQfD获得一组演示数据，并将其永久保留在回放缓冲区中。
2. **预训练**：DQfD在开始与环境交互之前，首先仅在演示数据上训练。
3. **监督损失**：除TD损失外，还应用大间隔监督损失，将演示者动作的值推高到其他动作值之上（Piot, Geist, and Pietquin 2014a）。
4. **L2正则化损失**：算法还对网络权重添加L2正则化损失，以防止在演示数据上过拟合。
5. **N步TD损失**：智能体用1步和n步回报的混合目标更新其Q网络。
6. **演示优先级奖励**：演示转移的优先级获得 $\epsilon_d$ 的加成，以提高其被采样的频率。

### 算法1：从演示中进行深度Q学习

```
输入：D_replay（用演示数据集初始化），θ（初始行为网络权重，随机），
      θ'（目标网络权重，随机），τ（目标网络更新频率），
      k（预训练梯度更新次数）

// 预训练阶段
for t = 1, 2, ..., k do
    从 D_replay 中按优先级采样小批量的 n 个转移
    使用目标网络计算损失 J(Q)
    执行梯度下降步骤以更新 θ
    if t mod τ = 0 then θ' ← θ end if
end for

// 在线学习阶段
for t = 1, 2, ... do
    从行为策略采样动作 a ~ π^ε_Qθ
    执行动作 a 并观察 (s', r)
    将 (s, a, r, s') 存入 D_replay，超容量时覆盖最旧的自生成转移
    从 D_replay 中按优先级采样小批量的 n 个转移
    使用目标网络计算损失 J(Q)
    执行梯度下降步骤以更新 θ
    if t mod τ = 0 then θ' ← θ end if
    s ← s'
end for
```

行为策略 $\pi^\epsilon_{Q_\theta}$ 为关于 $Q_\theta$ 的 $\epsilon$-贪心策略。

---

## 5 实验设置

我们在街机学习环境（Arcade Learning Environment, ALE）（Bellemare et al. 2013）上评估了DQfD。ALE是一组Atari游戏，是DQN的标准基准测试，包含许多人类表现仍优于最佳学习智能体的游戏。智能体从降采样为84x84、转换为灰度的游戏屏幕图像进行游戏，智能体将四帧叠加作为其状态。智能体必须为每个游戏输出18种可能动作之一。智能体使用0.99的折扣因子，所有动作重复四个Atari帧。每个回合用最多30个空操作初始化，以提供随机起始位置。报告的分数是Atari游戏中的分数，不论智能体内部如何表示奖励。

对于所有实验，我们评估了三种不同的算法，每种平均四次试验：

- **完整DQfD算法**（使用人类演示）
- **PDD DQN学习**（无任何演示数据）
- **监督模仿学习**（使用演示数据，无任何环境交互）

我们在六个Atari游戏上对所有算法进行了非正式的参数调优，然后对整套游戏使用相同的参数。我们对优先级和n步回报参数的粗略搜索为DQfD和PDD DQN得出了相同的最佳参数。PDD DQN与DQfD不同之处在于它没有演示数据、预训练、监督损失或正则化损失。我们在PDD DQN中包含了n步回报，以提供DQfD和PDD DQN之间更好的基线比较。三种算法都使用对偶状态-优势卷积网络架构（Dueling State-Advantage Convolutional Network Architecture）（Wang et al. 2016）。

对于监督模仿的比较，我们使用交叉熵损失对演示者的动作进行监督分类，使用与DQfD相同的网络架构和L2正则化。模仿算法不使用任何TD损失。模仿学习仅从预训练中学习，不进行任何额外交互。

我们在随机选择的42个Atari游戏子集上进行了实验。一位人类玩家每个游戏玩了3到12次。每个回合要么玩到游戏终止，要么玩20分钟。在游戏过程中，我们记录了智能体的状态、动作、奖励和终止信号。人类演示的范围从每个游戏5,574到75,472个转移。与其他类似工作相比，DQfD从非常小的数据集中学习，因为AlphaGo（Silver et al. 2016）从3000万个人类转移中学习，DQN（Mnih et al. 2015）从超过2亿帧中学习。DQfD较小的演示数据集使得在不过拟合的情况下学习良好表征更加困难。

我们发现，在许多人类玩家优于DQN的游戏中，原因是DQN将所有奖励裁剪为1进行训练。例如，在Private Eye中，DQN没有理由选择奖励25,000与奖励10的动作。为使人类演示者和智能体使用的奖励函数更一致，我们使用未裁剪的奖励并用对数尺度转换奖励：

$$r_{agent} = \text{sign}(r) \cdot \log(1 + |r|)$$

这种转换将奖励保持在神经网络可学习的合理范围内，同时传达了关于各个奖励相对大小的重要信息。这些调整后的奖励由实验中所有算法内部使用。结果仍使用实际游戏分数报告，如Atari文献中通常所做的那样（Mnih et al. 2015）。

---

## 6 结果

### 学习曲线

图1展示了三个游戏的学习曲线：Hero、Pitfall和Road Runner。

> **图1**：Hero、Pitfall和Road Runner游戏上各算法的在线得分。在Hero和Pitfall上，DQfD利用人类演示实现了高于任何先前发表结果的分数。最后一个子图展示了五个不同游戏中演示数据相对于均匀采样的超采样比率。
>
> - 三个子图分别显示Hero、Pitfall、Road Runner的训练回合回报曲线，比较DQfD、模仿学习和PDD DQN三种算法
> - 第四个子图显示Hero、Montezuma's Revenge、Pitfall、Q-Bert、Road Runner五个游戏的演示数据超采样比率

在Hero上，DQfD实现了高于任何人类演示以及任何先前发表结果的分数。Pitfall可能是最困难的Atari游戏，它具有非常稀疏的正奖励和密集的负奖励。先前没有方法在该游戏上获得任何正奖励，而DQfD在300万步周期内的最佳分数平均为394.0。

在Road Runner上，智能体通常通过与人类游戏方式大不相同的得分利用来学习超人类策略。我们的演示仅为人类水平，最高分数为20,200。Road Runner是人类演示集最小的游戏（仅5,574个转移）。尽管如此，DQfD在前3600万步仍然比PDD DQN获得更高的分数，之后与PDD DQN的性能持平。

图1右侧子图显示了演示数据被采样的频率相对于均匀采样的比率。对于最困难的游戏如Pitfall和Montezuma's Revenge，演示数据随时间被更频繁地采样。对于大多数其他游戏，该比率收敛到一个近乎恒定的水平，不同游戏有所不同。

### 初始性能

在现实任务中，智能体必须从第一个动作开始就表现良好且快速学习。DQfD在42个游戏中的41个游戏的前一百万步中性能优于PDD DQN。此外，在31个游戏中，DQfD的初始性能高于纯模仿学习，因为TD损失的加入帮助智能体更好地泛化演示数据。平均而言，PDD DQN直到任务进行到8300万步才超过DQfD的性能，且在平均分数上从未超越。

### 最先进结果

DQfD能够利用人类演示在最困难的Atari游戏上学到更好的策略。我们将DQfD在2亿步中的分数与其他深度强化学习方法进行了比较：DQN、Double DQN、Prioritized DQN、Dueling DQN、PopArt、DQN+CTS 和 DQN+PixelCNN（Mnih et al. 2015; van Hasselt, Guez, and Silver 2016; Schaul et al. 2016; Wang et al. 2016; van Hasselt et al. 2016; Ostrovski et al. 2017）。我们取4个种子平均的最佳300万步窗口作为DQfD的分数。DQfD在42个游戏中的11个游戏上取得了优于这些算法的分数，如表1所示。

> **表1**：DQfD在11个游戏上实现了高于任何先前发表的深度RL结果（使用随机空操作起始）的分数。先前结果取最佳智能体在最佳迭代时评估100个回合的结果。DQfD分数为4个种子平均的最佳300万步窗口，平均为508个回合。

| 游戏 | DQfD | 先前最佳 | 先前最佳算法 |
|------|------|---------|------------|
| Alien | 4745.9 | 4461.4 | Dueling DQN |
| Asteroids | 3796.4 | 2869.3 | PopArt |
| Atlantis | 920213.9 | 395762.0 | Prior. Dueling DQN |
| Battle Zone | 41971.7 | 37150.0 | Dueling DQN |
| Gravitar | 1693.2 | 859.1 | DQN+PixelCNN |
| Hero | 105929.4 | 23037.7 | Prioritized DQN |
| Montezuma Revenge | 4739.6 | 3705.5 | DQN+CTS |
| Pitfall | 50.8 | 0.0 | Prior. Dueling DQN |
| Private Eye | 40908.2 | 15806.5 | DQN+PixelCNN |
| Q-Bert | 21792.7 | 19220.3 | Dueling DQN |
| Up N Down | 82555.0 | 44939.6 | Dueling DQN |

注意：我们未与A3C（Mnih et al. 2016）或Reactor（Gruslys et al. 2017）比较，因为仅有人类起始的已发表结果；未与UNREAL（Jaderberg et al. 2016）比较，因为他们为每个游戏选择最佳超参数。尽管如此，DQfD仍在10个游戏上超越了UNREAL的最佳结果。基于计数的探索DQN（Ostrovski et al. 2017）专门设计用于最困难的探索游戏并在其上取得最佳结果。在两种算法都运行的六个稀疏奖励困难探索游戏中，DQfD在其中四个游戏中学到了更好的策略。

DQfD在42个游戏中的29个游戏中超越了其获得的最差演示回合，在14个游戏中学会了比最佳演示回合更好的表现。相比之下，纯模仿学习在每个游戏中都不如演示者的表现。

### 消融实验

图2展示了在两个DQfD实现最先进结果的游戏（Montezuma's Revenge和Q-Bert）上，将 $\lambda_1$ 和 $\lambda_2$ 设为0的DQfD的比较。

> **图2**：左侧子图显示了在Montezuma's Revenge和Q-Bert游戏上移除部分损失的DQfD在线奖励。移除任一损失都会降低算法性能。右侧子图将DQfD与相关工作部分的三种算法进行比较。其他方法的表现不如DQfD，尤其在Montezuma's Revenge上。
>
> - 左列两个子图：DQfD vs 无监督损失 vs 无n步TD损失，在Montezuma's Revenge和Q-Bert上
> - 右列两个子图：DQfD vs ADET vs Human Experience Replay vs Replay Buffer Spiking，在Montezuma's Revenge和Q-Bert上

正如预期，没有任何监督损失的预训练会导致网络朝未锚定的Q学习目标训练，智能体以更低的性能起步且改进更慢。移除n步TD损失对初始性能也有几乎同样大的影响，因为n步TD损失极大地帮助了从有限演示数据集中的学习。

### 与相关算法的比较

图2右侧子图将DQfD与三种利用演示数据的相关DQN算法进行比较：

- **回放缓冲区注入（RBS）**（Lipton et al. 2016）：即回放缓冲区初始填满演示数据的PDD DQN
- **人类经验回放（HER）**（Hosu and Rebedea 2016）：保留演示数据并在每个小批量中混合演示和智能体数据
- **带专家轨迹的加速DQN（ADET）**（Lakshminarayanan, Ozair, and Bengio 2016）：本质上是将大间隔监督损失替换为交叉熵损失的DQfD

结果表明，这三种方法在两个游戏中都不如DQfD。拥有监督损失对良好性能至关重要，DQfD和ADET的表现远优于其他两种算法。所有算法使用与DQfD完全相同的演示数据。我们在所有这些算法中都包含了优先回放机制和n步回报，以使比较尽可能公平。

---

## 7 讨论

我们在本文中提出的学习框架在数据中心控制、自动驾驶车辆（Hester and Stone 2013）或推荐系统（Shani, Heckerman, and Brafman 2005）等现实问题中非常常见。在这些问题中，通常没有精确的模拟器可用，学习必须在具有真实后果的真实系统上进行。然而，通常有系统由先前控制器操作的数据可用。我们提出了一种名为DQfD的新算法，利用这些数据来加速在真实系统上的学习。它首先仅在演示数据上预训练，使用1步TD、n步TD、监督和正则化损失的组合，使其具有合理的策略作为任务学习的良好起点。一旦开始与任务交互，它通过从自生成数据和演示数据中采样继续学习。每个小批量中两种数据的比例由优先回放机制自动控制。

我们展示了DQfD相比PDD DQN在初始性能上获得了巨大提升。DQfD在42个Atari游戏中的41个游戏的前一百万步中性能优于PDD DQN，平均DQN需要8200万步才能匹配DQfD的性能。在大多数现实任务中，智能体可能永远不会获得数亿步的学习机会。我们还表明DQfD优于其他三种在强化学习中利用演示数据的算法。DQfD优于所有这些算法的事实清楚地表明，在有这类演示数据可用的任何现实强化学习应用中，它是更好的选择。

除了早期性能提升外，DQfD还能利用人类演示在11个Atari游戏上实现最先进的结果。其中许多是最困难的探索游戏（如Montezuma's Revenge、Pitfall、Private Eye），演示数据可以替代更智能的探索。这一结果使得将强化学习部署到原本需要更智能探索的问题成为可能。

DQfD在每个游戏仅有非常少量的演示数据（5,574到75,472个转移）的情况下实现了这些结果，这些数据仅需几分钟的游戏即可轻松生成。DQN和DQfD接收的用于强化学习的交互数据比演示数据多三个数量级。DQfD展示了用正确的算法添加少量演示数据所能实现的收益。正如相关工作比较所示，天真地将这少量数据添加到纯深度强化学习算法中（例如仅预训练或填充回放缓冲区）不能提供类似的收益，有时甚至会产生不利影响。

这些结果可能看起来很明显，因为DQfD可以访问特权数据，但奖励和演示在数学上是不同的训练信号，天真地组合它们可能导致灾难性结果。仅在人类演示上进行监督学习是不成功的，而DQfD在42个游戏中的14个游戏中学会了超越最佳演示。DQfD还优于三种先前将演示数据融入DQN的算法。我们认为，预训练期间所有四种损失的组合对于智能体学习一个不会被预训练后训练信号切换所破坏的连贯表征至关重要。即使在预训练之后，智能体也必须继续使用专家数据。特别是，图1的右子图表明，对于最困难的探索游戏，所需的专家数据比例（由优先回放选择）在交互阶段增长，因为当智能体到达游戏中的新画面时，演示数据变得更加有用。RBS展示了一个仅在初始拥有演示数据不足以提供良好性能的例子。

从人类演示中学习尤其困难。在大多数游戏中，模仿学习甚至无法在演示数据集上完美分类演示者的动作。人类可能以与智能体学到的策略大不相同的方式玩游戏，并且可能使用智能体状态表示中不可用的信息。在未来的工作中，我们计划测量演示和智能体数据之间的这些差异，以便提出从演示中获取更多价值的方法。另一个未来方向是将这些概念应用于连续动作空间的领域，其中分类损失变为回归损失。

---

## 致谢

作者感谢 Keith Anderson、Chris Apps、Ben Coppin、Joe Fenton、Nando de Freitas、Chris Gamble、Thore Graepel、Georg Ostrovski、Cosmin Paduraru、Jack Rae、Amir Sadik、Jon Scholz、David Silver、Toby Pohlen、Tom Stepleton、Ziyu Wang 以及 DeepMind 的许多同事提供的深刻讨论、代码贡献和其他帮助。

---

## 参考文献

[Abbeel et al. 2007] Abbeel, P.; Coates, A.; Quigley, M.; and Ng, A. Y. 2007. An application of reinforcement learning to aerobatic helicopter flight. In Advances in Neural Information Processing Systems (NIPS).

[Bellemare et al. 2013] Bellemare, M. G.; Naddaf, Y.; Veness, J.; and Bowling, M. 2013. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research (JAIR) 47:253–279.

[Brys et al. 2015] Brys, T.; Harutyunyan, A.; Suay, H.; Chernova, S.; Taylor, M.; and Nowé, A. 2015. Reinforcement learning from demonstration through shaping. In International Joint Conference on Artificial Intelligence (IJCAI).

[Cederborg et al. 2015] Cederborg, T.; Grover, I.; Isbell, C.; and Thomaz, A. 2015. Policy shaping with human teachers. In International Joint Conference on Artificial Intelligence (IJCAI 2015).

[Chemali and Lezaric 2015] Chemali, J., and Lezaric, A. 2015. Direct policy iteration from demonstrations. In International Joint Conference on Artificial Intelligence (IJCAI).

[Duan et al. 2017] Duan, Y.; Andrychowicz, M.; Stadie, B. C.; Ho, J.; Schneider, J.; Sutskever, I.; Abbeel, P.; and Zaremba, W. 2017. One-shot imitation learning. CoRR abs/1703.07326.

[Finn, Levine, and Abbeel 2016] Finn, C.; Levine, S.; and Abbeel, P. 2016. Guided cost learning: Deep inverse optimal control via policy optimization. In International Conference on Machine Learning (ICML).

[Gruslys et al. 2017] Gruslys, A.; Gheshlaghi Azar, M.; Bellemare, M. G.; and Munos, R. 2017. The Reactor: A Sample-Efficient Actor-Critic Architecture. ArXiv e-prints.

[Hester and Stone 2013] Hester, T., and Stone, P. 2013. TEXPLORE: Real-time sample-efficient reinforcement learning for robots. Machine Learning 90(3).

[Ho and Ermon 2016] Ho, J., and Ermon, S. 2016. Generative adversarial imitation learning. In Advances in Neural Information Processing Systems (NIPS).

[Hosu and Rebedea 2016] Hosu, I.-A., and Rebedea, T. 2016. Playing atari games with deep reinforcement learning and human checkpoint replay. In ECAI Workshop on Evaluating General Purpose AI.

[Jaderberg et al. 2016] Jaderberg, M.; Mnih, V.; Czarnecki, W. M.; Schaul, T.; Leibo, J. Z.; Silver, D.; and Kavukcuoglu, K. 2016. Reinforcement learning with unsupervised auxiliary tasks. CoRR abs/1611.05397.

[Kim et al. 2013] Kim, B.; Farahmand, A.; Pineau, J.; and Precup, D. 2013. Learning from limited demonstrations. In Advances in Neural Information Processing Systems (NIPS).

[Lakshminarayanan, Ozair, and Bengio 2016] Lakshminarayanan, A. S.; Ozair, S.; and Bengio, Y. 2016. Reinforcement learning with few expert demonstrations. In NIPS Workshop on Deep Learning for Action and Interaction.

[LeCun, Bengio, and Hinton 2015] LeCun, Y.; Bengio, Y.; and Hinton, G. 2015. Deep learning. Nature 521(7553):436–444.

[Levine et al. 2016] Levine, S.; Finn, C.; Darrell, T.; and Abbeel, P. 2016. End-to-end training of deep visuomotor policies. Journal of Machine Learning (JMLR) 17:1–40.

[Lipton et al. 2016] Lipton, Z. C.; Gao, J.; Li, L.; Li, X.; Ahmed, F.; and Deng, L. 2016. Efficient exploration for dialog policy learning with deep BBQ network & replay buffer spiking. CoRR abs/1608.05081.

[Mnih et al. 2015] Mnih, V.; Kavukcuoglu, K.; Silver, D.; Rusu, A. A.; Veness, J.; Bellemare, M. G.; Graves, A.; Riedmiller, M.; Fidjeland, A. K.; Ostrovski, G.; Petersen, S.; Beattie, C.; Sadik, A.; Antonoglou, I.; King, H.; Kumaran, D.; Wierstra, D.; Legg, S.; and Hassabis, D. 2015. Human-level control through deep reinforcement learning. Nature 518(7540):529–533.

[Mnih et al. 2016] Mnih, V.; Badia, A. P.; Mirza, M.; Graves, A.; Lillicrap, T.; Harley, T.; Silver, D.; and Kavukcuoglu, K. 2016. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, 1928–1937.

[Ostrovski et al. 2017] Ostrovski, G.; Bellemare, M. G.; van den Oord, A.; and Munos, R. 2017. Count-based exploration with neural density models. CoRR abs/1703.01310.

[Piot, Geist, and Pietquin 2014a] Piot, B.; Geist, M.; and Pietquin, O. 2014a. Boosted bellman residual minimization handling expert demonstrations. In European Conference on Machine Learning (ECML).

[Piot, Geist, and Pietquin 2014b] Piot, B.; Geist, M.; and Pietquin, O. 2014b. Boosted and Reward-regularized Classification for Apprenticeship Learning. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS).

[Ross, Gordon, and Bagnell 2011] Ross, S.; Gordon, G. J.; and Bagnell, J. A. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics (AISTATS).

[Schaal 1996] Schaal, S. 1996. Learning from demonstration. In Advances in Neural Information Processing Systems (NIPS).

[Schaul et al. 2016] Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2016. Prioritized experience replay. In Proceedings of the International Conference on Learning Representations, volume abs/1511.05952.

[Shani, Heckerman, and Brafman 2005] Shani, G.; Heckerman, D.; and Brafman, R. I. 2005. An mdp-based recommender system. Journal of Machine Learning Research 6:1265–1295.

[Silver et al. 2016] Silver, D.; Huang, A.; Maddison, C. J.; Guez, A.; Sifre, L.; van den Driessche, G.; Schrittwieser, J.; Antonoglou, I.; Panneershelvam, V.; Lanctot, M.; Dieleman, S.; Grewe, D.; Nham, J.; Kalchbrenner, N.; Sutskever, I.; Lillicrap, T.; Leach, M.; Kavukcuoglu, K.; Graepel, T.; and Hassabis, D. 2016. Mastering the game of Go with deep neural networks and tree search. Nature 529:484–489.

[Suay et al. 2016] Suay, H. B.; Brys, T.; Taylor, M. E.; and Chernova, S. 2016. Learning from demonstration for shaping through inverse reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS).

[Subramanian, Jr., and Thomaz 2016] Subramanian, K.; Jr., C. L. I.; and Thomaz, A. 2016. Exploration from demonstration for interactive reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS).

[Sun et al. 2017] Sun, W.; Venkatraman, A.; Gordon, G. J.; Boots, B.; and Bagnell, J. A. 2017. Deeply aggrevated: Differentiable imitation learning for sequential prediction. CoRR abs/1703.01030.

[Sutton and Barto 1998] Sutton, R. S., and Barto, A. G. 1998. Introduction to reinforcement learning. MIT Press.

[Syed and Schapire 2007] Syed, U., and Schapire, R. E. 2007. A game-theoretic approach to apprenticeship learning. In Advances in Neural Information Processing Systems (NIPS).

[Syed, Bowling, and Schapire 2008] Syed, U.; Bowling, M.; and Schapire, R. E. 2008. Apprenticeship learning using linear programming. In International Conference on Machine Learning (ICML).

[Taylor, Suay, and Chernova 2011] Taylor, M.; Suay, H.; and Chernova, S. 2011. Integrating reinforcement learning with human demonstrations of varying ability. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS).

[van Hasselt et al. 2016] van Hasselt, H. P.; Guez, A.; Hessel, M.; Mnih, V.; and Silver, D. 2016. Learning values across many orders of magnitude. In Advances in Neural Information Processing Systems (NIPS).

[van Hasselt, Guez, and Silver 2016] van Hasselt, H.; Guez, A.; and Silver, D. 2016. Deep reinforcement learning with double Q-learning. In AAAI Conference on Artificial Intelligence (AAAI).

[Wang et al. 2016] Wang, Z.; Schaul, T.; Hessel, M.; van Hasselt, H.; Lanctot, M.; and de Freitas, N. 2016. Dueling network architectures for deep reinforcement learning. In International Conference on Machine Learning (ICML).

[Watter et al. 2015] Watter, M.; Springenberg, J. T.; Boedecker, J.; and Riedmiller, M. A. 2015. Embed to control: A locally linear latent dynamics model for control from raw images. In Advances in Neural Information Processing (NIPS).

---

## 附录：超参数

以下为三种算法使用的参数。DQfD使用所有参数，其他两种算法仅使用适用的参数。

- 预训练步数 $k = 750,000$ 次小批量更新
- N步回报权重 $\lambda_1 = 1.0$
- 监督损失权重 $\lambda_2 = 1.0$
- L2正则化权重 $\lambda_3 = 10^{-5}$
- 专家间隔 $l(a_E, a)$（当 $a \neq a_E$ 时）$= 0.8$
- $\epsilon$-贪心探索，$\epsilon = 0.01$，与Double DQN相同（van Hasselt, Guez, and Silver 2016）
- 优先回放指数 $\alpha = 0.4$
- 优先回放常数 $\epsilon_a = 0.001$，$\epsilon_d = 1.0$
- 优先回放重要性采样指数 $\beta_0 = 0.6$，如 Schaul et al. (2016)
- N步回报 $n = 10$
- 目标网络更新周期 $\tau = 10,000$，如 Mnih et al. (2015)

---

## 附录：全部42个游戏学习曲线

> 附录包含所有42个Atari游戏的学习曲线图，每张图展示DQfD、模仿学习和PDD DQN三种算法在200个训练迭代中的回合回报对比。游戏包括：Alien, Amidar, Assault, Asterix, Asteroids, Atlantis, Bank Heist, Battle Zone, Beam Rider, Bowling, Boxing, Breakout, Chopper Command, Crazy Climber, Defender, Demon Attack, Double Dunk, Enduro, Fishing Derby, Freeway, Gopher, Gravitar, Hero, Ice Hockey, James Bond, Kangaroo, Krull, Kung Fu Master, Montezuma Revenge, Ms Pacman, Name This Game, Pitfall, Pong, Private Eye, Q-Bert, Riverraid, Road Runner, Seaquest, Space Invaders, Star Gunner, Tutankham, Up N Down。
