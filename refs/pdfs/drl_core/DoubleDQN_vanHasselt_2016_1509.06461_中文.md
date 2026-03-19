# 基于双重Q学习的深度强化学习

**作者：** Hado van Hasselt, Arthur Guez, David Silver

**机构：** Google DeepMind

---

## 摘要

众所周知，流行的Q学习（Q-learning）算法在某些条件下会过高估计动作值。此前尚不清楚这种过高估计在实践中是否常见、是否会损害性能，以及是否能普遍加以预防。本文对上述所有问题给出了肯定的回答。具体而言，我们首先表明，最近提出的将Q学习与深度神经网络相结合的DQN算法，在Atari 2600领域的某些游戏中存在严重的过高估计问题。随后我们证明，最初在表格型（tabular）环境中提出的双重Q学习（Double Q-learning）算法背后的思想，可以推广到大规模函数逼近（function approximation）的场景中。我们提出了一种针对DQN算法的具体改进方案，并表明所得算法不仅如假设的那样减少了观测到的过高估计，而且在多个游戏上带来了显著更好的性能表现。

---

## 引言

强化学习（reinforcement learning）（Sutton and Barto, 1998）的目标是通过优化累积未来奖励信号，为序贯决策问题学习好的策略（policy）。Q学习（Watkins, 1989）是最流行的强化学习算法之一，但已知它有时会学到不切实际的高动作值，原因在于它包含一个对估计动作值取最大值的步骤，该步骤倾向于偏好被过高估计的值而非被过低估计的值。

在先前的工作中，过高估计被归因于函数逼近不够灵活（Thrun and Schwartz, 1993）以及噪声（van Hasselt, 2010, 2011）。本文统一了这些观点，并表明当动作值不精确时——无论逼近误差来源如何——过高估计都可能发生。当然，在学习过程中不精确的值估计是常态，这意味着过高估计可能比此前认识到的要普遍得多。

一个悬而未决的问题是：如果过高估计确实发生了，这在实践中是否会对性能产生负面影响？过于乐观的值估计本身不一定是问题。如果所有值都被均匀地提高，则动作之间的相对偏好得以保持，我们不会期望由此导致的策略变得更差。此外，已知有时乐观是有益的：面对不确定性时保持乐观是一种广为人知的探索技术（Kaelbling et al., 1996）。然而，如果过高估计不是均匀的，也不集中在我们希望深入了解的状态上，则它们可能对所得策略的质量产生负面影响。Thrun and Schwartz (1993) 给出了具体的例子，表明这种情况甚至会在渐近意义下导致次优策略。

为了验证过高估计在实践中和大规模场景下是否会发生，我们研究了最近提出的DQN算法（Mnih et al., 2015）的性能。DQN将Q学习与灵活的深度神经网络结合，并在大量多样的确定性Atari 2600游戏上进行了测试，在许多游戏上达到了人类水平的性能。从某种意义上说，这一设定对Q学习来说是一个最佳情况：深度神经网络提供了灵活的函数逼近能力，具有较低的渐近逼近误差的潜力，而环境的确定性则避免了噪声的有害影响。然而令人意外的是，我们发现即使在这种相对有利的设定下，DQN有时也会大幅过高估计动作的值。

我们表明，最初在表格型环境中提出的双重Q学习算法（van Hasselt, 2010）背后的思想，可以推广到任意函数逼近中使用，包括深度神经网络。基于此，我们构建了一种新算法，称为**双重DQN（Double DQN）**。随后我们证明，该算法不仅产生了更准确的值估计，还在多个游戏上取得了更高的分数。这表明DQN的过高估计确实导致了较差的策略，而减少这些过高估计是有益的。此外，通过改进DQN，我们在Atari领域取得了当时最优的结果。

---

## 背景

为了解决序贯决策问题，我们可以学习每个动作的最优值估计，即在该状态下采取该动作并此后遵循最优策略所获得的期望未来奖励总和。在给定策略 $\pi$ 下，动作 $a$ 在状态 $s$ 中的真实值为：

$$Q^\pi(s, a) \equiv \mathbb{E}[R_1 + \gamma R_2 + \ldots \mid S_0 = s, A_0 = a, \pi]$$

其中 $\gamma \in [0, 1]$ 是折扣因子（discount factor），用于权衡即时奖励与未来奖励的重要性。最优值则为 $Q^*(s, a) = \max_\pi Q^\pi(s, a)$。最优策略可以通过在每个状态中选择最高值的动作轻松得出。

最优动作值的估计可以通过Q学习（Watkins, 1989）来学习，这是时序差分学习（temporal difference learning）（Sutton, 1988）的一种形式。大多数有趣的问题规模太大，无法对所有状态中的所有动作值分别学习。取而代之的是，我们可以学习一个参数化的值函数（parameterized value function）$Q(s, a; \theta_t)$。在状态 $S_t$ 中采取动作 $A_t$，观测到即时奖励 $R_{t+1}$ 和后继状态 $S_{t+1}$ 后，标准Q学习的参数更新规则为：

$$\theta_{t+1} = \theta_t + \alpha(Y^Q_t - Q(S_t, A_t; \theta_t))\nabla_{\theta_t}Q(S_t, A_t; \theta_t) \tag{1}$$

其中 $\alpha$ 是标量步长，目标值（target） $Y^Q_t$ 定义为：

$$Y^Q_t \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t) \tag{2}$$

该更新类似于随机梯度下降（stochastic gradient descent），将当前值 $Q(S_t, A_t; \theta_t)$ 向目标值 $Y^Q_t$ 方向更新。

### 深度Q网络（Deep Q Networks）

深度Q网络（DQN）是一种多层神经网络，对于给定状态 $s$ 输出一个动作值向量 $Q(s, \cdot; \theta)$，其中 $\theta$ 为网络参数。对于 $n$ 维状态空间和包含 $m$ 个动作的动作空间，该神经网络是一个从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的函数。Mnih et al. (2015) 提出的DQN算法有两个重要组成部分：目标网络（target network）的使用和经验回放（experience replay）。目标网络的参数为 $\theta^-$，其结构与在线网络（online network）相同，但参数每隔 $\tau$ 步从在线网络复制一次，即此时 $\theta^-_t = \theta_t$，在其他步骤保持固定。DQN使用的目标为：

$$Y^{DQN}_t \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-_t) \tag{3}$$

对于经验回放（Lin, 1992），观测到的转移（transition）被存储一段时间，并从该记忆库中均匀采样来更新网络。目标网络和经验回放都极大地提升了算法的性能（Mnih et al., 2015）。

### 双重Q学习（Double Q-learning）

标准Q学习和DQN中的最大值算子（max operator），如式 (2) 和式 (3) 所示，使用相同的值既来选择动作也来评估动作。这使得选择被过高估计的值的可能性更大，从而导致过于乐观的值估计。为防止这一问题，我们可以将选择与评估解耦（decouple）。这就是双重Q学习（van Hasselt, 2010）背后的思想。

在原始的双重Q学习算法中，通过将每次经验随机分配给两个值函数之一来学习两个值函数，从而有两组权重 $\theta$ 和 $\theta'$。每次更新时，一组权重用于确定贪心策略（greedy policy），另一组用于确定该策略的值。为了便于清晰比较，我们可以先将Q学习中的选择和评估分离开来，将其目标式 (2) 重写为：

$$Y^Q_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t)$$

双重Q学习的目标则可以写为：

$$Y^{DoubleQ}_t \equiv R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta'_t) \tag{4}$$

注意，动作的选择（在 $\arg\max$ 中）仍然使用在线权重 $\theta_t$。这意味着与Q学习一样，我们仍然在估计由 $\theta_t$ 定义的当前值所对应的贪心策略的值。然而，我们使用第二组权重 $\theta'_t$ 来公正地评估该策略的值。通过交换 $\theta$ 和 $\theta'$ 的角色，可以对称地更新第二组权重。

---

## 估计误差导致的过度乐观

Q学习的过高估计最先由Thrun and Schwartz (1993) 研究，他们证明如果动作值包含在区间 $[-\epsilon, \epsilon]$ 内均匀分布的随机误差，则每个目标的过高估计量上界为 $\gamma\epsilon\frac{m-1}{m+1}$，其中 $m$ 是动作数量。此外，Thrun and Schwartz 给出了一个具体例子，说明这些过高估计甚至在渐近情况下也会导致次优策略，并展示了在使用函数逼近时过高估计在一个小型玩具问题中的表现。后来 van Hasselt (2010) 论证了即使使用表格表示（tabular representation），环境中的噪声也会导致过高估计，并提出了双重Q学习作为解决方案。

在本节中，我们更一般地证明了任何类型的估计误差都可以引发向上偏差（upward bias），无论这些误差是由环境噪声、函数逼近、非平稳性还是其他来源引起的。这一点很重要，因为在实践中任何方法在学习过程中都会产生一些不准确性，仅仅因为真实值最初是未知的。

上述Thrun and Schwartz (1993) 的结果给出了特定设定下过高估计的上界，但也可以推导出一个下界——这可能更为有趣。

**定理 1.** 考虑一个状态 $s$，其中所有真实最优动作值相等，即对某个 $V^*(s)$ 有 $Q^*(s, a) = V^*(s)$。设 $Q_t$ 为任意值估计，总体上是无偏的，即 $\sum_a (Q_t(s, a) - V^*(s)) = 0$，但并非全部正确，使得 $\frac{1}{m}\sum_a (Q_t(s, a) - V^*(s))^2 = C$，其中 $C > 0$，$m \geq 2$ 为状态 $s$ 中的动作数量。在这些条件下：

$$\max_a Q_t(s, a) \geq V^*(s) + \sqrt{\frac{C}{m-1}}$$

此下界是紧的（tight）。在相同条件下，双重Q学习估计的绝对误差的下界为零。（证明见附录。）

注意，我们并不需要假设不同动作的估计误差是独立的。该定理表明，即使值估计在平均意义上是正确的，来自任何来源的估计误差都可以将估计值推高，使其偏离真实最优值。

定理 1 中的下界随动作数量的增加而减小。这是考虑下界所导致的，因为下界的取得需要非常特定的值配置。更典型的情况是，过度乐观随动作数量的增加而增大，如图 1 所示。

> **图 1：** 橙色柱表示当动作值 $Q(s, a) = V^*(s) + \epsilon_a$，且误差 $\{\epsilon_a\}_{a=1}^m$ 为独立标准正态随机变量时，单次Q学习更新中的偏差。第二组动作值 $Q'$ 以相同且独立的方式生成，用于蓝色柱。所有柱状图为100次重复的平均值。Q学习的过高估计随动作数量增大，而双重Q学习则是无偏的。

另一个例子：如果对所有动作有 $Q^*(s, a) = V^*(s)$，且估计误差 $Q_t(s, a) - V^*(s)$ 在 $[-1, 1]$ 上均匀随机分布，则过度乐观为 $\frac{m-1}{m+1}$（证明见附录）。

接下来我们转向函数逼近的情况，考虑一个实值连续状态空间，每个状态有10个离散动作。为简单起见，此例中真实最优动作值仅依赖于状态，即每个状态中所有动作具有相同的真实值。

> **图 2：** 学习过程中过高估计的示意图。每个状态（x轴）有10个动作。左列显示真实值 $V^*(s)$（紫色线），所有真实动作值由 $Q^*(s, a) = V^*(s)$ 定义。绿色线显示一个动作的估计值 $Q(s, a)$ 关于状态的函数，基于若干采样状态（绿色点）拟合真实值。中间列图显示所有估计值（绿色）及其最大值（黑色虚线）。最大值几乎在所有位置都高于真实值（左图中的紫色线）。右列图以橙色显示差异。右图中的蓝色线是双重Q学习使用第二组样本得到的估计。蓝色线更接近零，表明偏差更小。三行分别对应不同的真实函数（左，紫色）或拟合函数的容量（左，绿色）。

图 2 的三行展示了同一实验的不同变体。第一行和第二行的区别在于真实值函数不同，说明过高估计不是某个特定真实值函数的产物。第二行和第三行的区别在于函数逼近的灵活性。在左中图中，由于函数不够灵活，估计值甚至在某些采样状态上也不正确。底行的函数更灵活，但这在未见状态上导致更高的估计误差，从而带来更高的过高估计。这一点很重要，因为灵活的参数化函数逼近器在强化学习中被广泛使用（如 Tesauro 1995; Sallans and Hinton 2004; Riedmiller 2005; Mnih et al. 2015）。

与 van Hasselt (2010) 不同，我们没有使用统计论证来发现过高估计——图 2 的生成过程是完全确定性的。与 Thrun and Schwartz (1993) 不同，我们没有依赖具有不可约渐近误差的不灵活函数逼近；底行表明，一个足够灵活以覆盖所有样本的函数会导致高过高估计。这表明过高估计可以相当普遍地发生。

在上述例子中，即使假设我们在某些状态拥有真实动作值的样本，过高估计仍然会发生。如果我们对已经过于乐观的动作值进行自举（bootstrap），值估计会进一步恶化，因为这会导致过高估计在整个估计中传播。虽然均匀地过高估计值可能不会损害所得策略，但在实践中不同状态和动作的过高估计误差是不同的。过高估计与自举的结合会产生有害效应：传播了关于哪些状态比其他状态更有价值的错误相对信息，直接影响所学策略的质量。

这些过高估计不应与面对不确定性时的乐观（optimism in the face of uncertainty）（Sutton, 1990; Agrawal, 1995; Kaelbling et al., 1996; Auer et al., 2002; Brafman and Tennenholtz, 2003; Szita and Lorincz, 2008; Strehl et al., 2009）混淆，后者是对具有不确定值的状态或动作给予探索奖励。相反，此处讨论的过高估计仅在更新之后发生，导致在表面确定的情况下产生过度乐观。Thrun and Schwartz (1993) 已经指出，与面对不确定性时的乐观不同，这些过高估计实际上会阻碍最优策略的学习。我们将在后续实验中确认这种对策略质量的负面影响：当我们使用双重Q学习减少过高估计时，策略得到了改善。

---

## 双重DQN（Double DQN）

双重Q学习的思想是通过将目标中的最大值操作分解为动作选择和动作评估来减少过高估计。虽然未完全解耦，但DQN架构中的目标网络提供了第二个值函数的天然候选，无需引入额外的网络。因此，我们提出根据在线网络评估贪心策略，但使用目标网络来估计该策略的值。参照双重Q学习和DQN的命名，我们将所得算法称为**双重DQN**。其更新规则与DQN相同，但将目标 $Y^{DQN}_t$ 替换为：

$$Y^{DoubleDQN}_t \equiv R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta^-_t)$$

与双重Q学习式 (4) 相比，第二个网络的权重 $\theta'_t$ 被替换为目标网络的权重 $\theta^-_t$，用于评估当前贪心策略。目标网络的更新保持与DQN相同，仍然是定期复制在线网络。

这个版本的双重DQN可能是从DQN向双重Q学习方向的最小改动。其目标是在保持DQN算法其余部分不变以便公平比较的前提下，以最小的计算开销获得双重Q学习的大部分收益。

---

## 实验结果

在本节中，我们分析DQN的过高估计，并展示双重DQN在值准确性和策略质量方面对DQN的改进。为进一步测试方法的鲁棒性，我们还使用Nair et al. (2015) 提出的从人类专家轨迹生成的随机起始点来评估算法。

我们的测试平台为Atari 2600游戏，使用街机学习环境（Arcade Learning Environment）（Bellemare et al., 2013）。目标是让单一算法使用固定的超参数集合，仅从屏幕像素输入出发，通过交互分别学习每个游戏。这是一个严格的测试平台：不仅输入是高维的，而且不同游戏之间的视觉效果和游戏机制差异很大。好的解决方案必须严重依赖学习算法——仅靠调参来过拟合领域是不现实的。

我们严格遵循Mnih et al. (2015) 概述的实验设定和网络架构。简而言之，网络架构是一个卷积神经网络（convolutional neural network）（Fukushima, 1988; LeCun et al., 1998），包含3个卷积层和一个全连接隐藏层（参数总量约150万）。网络以最近4帧作为输入，输出每个动作的动作值。在每个游戏上，网络在单个GPU上训练2亿帧，大约需要1周时间。

### 过度乐观的结果

图 3 展示了DQN在六个Atari游戏中过高估计的例子。DQN和双重DQN均在Mnih et al. (2015) 描述的完全相同的条件下训练。

> **图 3：** 上两行显示DQN（橙色）和双重DQN（蓝色）在六个Atari游戏上的值估计。结果由使用Mnih et al. (2015) 超参数的6个不同随机种子运行DQN和双重DQN获得。深色线为种子间的中位数，阴影区域为10%和90%分位数（线性插值）。上行中水平的橙色线（DQN）和蓝色线（双重DQN）是在学习结束后运行对应智能体、从每个访问状态平均实际折扣回报得到的。如果不存在偏差，这些水平线应与图右侧的学习曲线吻合。中间行以对数尺度显示两个DQN过度乐观极端的游戏的值估计。底行显示过高估计对训练中评估分数的有害影响：当过高估计开始时分数下降。双重DQN的学习更加稳定。

DQN对当前贪心策略的值一贯地且有时大幅地过于乐观。相比之下，双重DQN的学习曲线更接近表示最终策略真实值的直线。注意蓝色直线通常高于橙色直线，这表明双重DQN不仅产生了更准确的值估计，还产生了更好的策略。

在Asterix和Wizard of Wor两个游戏中（注意y轴使用对数尺度），DQN出现了极端的过高估计和不稳定性。底行相应的分数图显示，DQN值估计的增长与分数的下降同时发生。虽然乍看之下可能会认为这种不稳定性与带函数逼近的离策略学习（off-policy learning）的固有不稳定性有关（Baird, 1995; Tsitsiklis and Van Roy, 1997; Sutton et al., 2008; Maei, 2011; Sutton et al., 2015），但双重DQN在相同条件下学习更加稳定，表明这些不稳定性的根源实际上是Q学习的过度乐观。图 3 仅展示了部分例子，但在所有49个测试的Atari游戏中都观察到了DQN的过高估计，只是程度不同。

### 所学策略的质量

过度乐观并不总是对所学策略的质量产生不利影响。例如，DQN在Pong游戏中尽管略有过高估计策略值，但仍然达到了最优行为。然而，减少过高估计可以显著提升学习的稳定性。表 1 总结了在无操作评估条件下（5分钟模拟器时间）49个游戏上的归一化性能。

**表 1：** 49个游戏上5分钟游玩的归一化性能总结。DQN结果来自 Mnih et al. (2015)。

|  | DQN | Double DQN |
|---|---|---|
| 中位数 | 93.5% | 114.7% |
| 平均值 | 241.1% | 330.3% |

如Mnih et al. (2015) 所述，每个评估回合（episode）开始时执行一种特殊的空操作（no-op），最多30次，为智能体提供不同的起始点。评估时使用一定的探索提供额外的随机性。双重DQN使用与DQN完全相同的超参数，以实现聚焦于减少过高估计的对照实验。所学策略在5分钟的模拟器时间（18000帧）内使用 $\epsilon$-贪心策略（$\epsilon = 0.05$）进行评估，分数为100个回合的平均值。双重DQN与DQN的唯一区别在于目标，使用 $Y^{DoubleDQN}_t$ 替代 $Y^{DQN}_t$。

为了跨游戏获得汇总统计数据，我们对每个游戏的分数进行如下归一化：

$$score_{normalized} = \frac{score_{agent} - score_{random}}{score_{human} - score_{random}} \tag{5}$$

详细比较显示，在多个游戏中双重DQN相比DQN有大幅提升。值得注意的例子包括：Road Runner（从233%提升至617%）、Asterix（从70%提升至180%）、Zaxxon（从54%提升至111%）以及Double Dunk（从17%提升至397%）。

### 对人类起始点的鲁棒性

先前评估的一个问题是，在具有唯一起始点的确定性游戏中，学习器可能只是记住了动作序列而无需太多泛化。通过从不同起始点测试智能体，可以检验所找到的解决方案是否具有良好的泛化能力（Nair et al., 2015）。

我们从人类专家的轨迹中为每个游戏采样了100个起始点。从每个起始点开始评估回合，运行模拟器最多108000帧（60Hz下30分钟）。智能体仅在起始点之后累积的奖励上进行评估。

对于调优版本的双重DQN，我们将目标网络复制间隔从10000帧增加到30000帧，将训练中的探索从 $\epsilon = 0.1$ 降低到 $\epsilon = 0.01$，评估时使用 $\epsilon = 0.001$。此外，调优版本在网络顶层对所有动作值使用单一共享偏置。这些改动各自都改善了性能，组合起来带来了明显更好的结果。

**表 2：** 49个游戏上30分钟人类起始点游玩的归一化性能总结。DQN结果来自 Nair et al. (2015)。

|  | DQN | Double DQN | Double DQN（调优） |
|---|---|---|---|
| 中位数 | 47.5% | 88.4% | 116.7% |
| 平均值 | 122.0% | 273.1% | 475.2% |

> **图 4：** 57个Atari游戏上的归一化分数，每个游戏使用人类起始点测试100个回合。与Mnih et al. (2015) 相比，额外测试了8个游戏（以星号和粗体标注）。图中显示了DQN、双重DQN和调优双重DQN在每个游戏上的归一化分数条形图。

双重DQN在这一更具挑战性的评估中表现出更强的鲁棒性，表明发生了适当的泛化，所找到的解决方案并未利用环境的确定性。这是令人鼓舞的，因为它表明朝着找到通用解决方案的方向取得了进展，而非找到一个不够鲁棒的确定性步骤序列。

---

## 讨论

本文有五项贡献。第一，我们说明了为什么Q学习在大规模问题中——即使这些问题是确定性的——也可能由于学习固有的估计误差而产生过度乐观。第二，通过分析Atari游戏上的值估计，我们展示了这些过高估计在实践中比此前认识到的更为普遍和严重。第三，我们证明了双重Q学习可以在大规模场景中成功减少这种过度乐观，从而带来更稳定可靠的学习。第四，我们提出了一种名为双重DQN的具体实现方案，该方案使用DQN算法已有的架构和深度神经网络，无需额外的网络或参数。最后，我们展示了双重DQN能够找到更好的策略，在Atari 2600领域取得了当时最优的结果。

---

## 致谢

我们感谢Tom Schaul、Volodymyr Mnih、Marc Bellemare、Thomas Degris、Georg Ostrovski和Richard Sutton提供的有益意见，感谢Google DeepMind全体成员营造的建设性研究环境。

---

## 参考文献

R. Agrawal. Sample mean based index policies with O(log n) regret for the multi-armed bandit problem. Advances in Applied Probability, pages 1054–1078, 1995.

P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3):235–256, 2002.

L. Baird. Residual algorithms: Reinforcement learning with function approximation. In Machine Learning: Proceedings of the Twelfth International Conference, pages 30–37, 1995.

M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The arcade learning environment: An evaluation platform for general agents. J. Artif. Intell. Res. (JAIR), 47:253–279, 2013.

R. I. Brafman and M. Tennenholtz. R-max-a general polynomial time algorithm for near-optimal reinforcement learning. The Journal of Machine Learning Research, 3:213–231, 2003.

K. Fukushima. Neocognitron: A hierarchical neural network capable of visual pattern recognition. Neural networks, 1(2):119–130, 1988.

L. P. Kaelbling, M. L. Littman, and A. W. Moore. Reinforcement learning: A survey. Journal of Artificial Intelligence Research, 4:237–285, 1996.

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

L. Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3):293–321, 1992.

H. R. Maei. Gradient temporal-difference learning algorithms. PhD thesis, University of Alberta, 2011.

V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

A. Nair, P. Srinivasan, S. Blackwell, C. Alcicek, R. Fearon, A. D. Maria, V. Panneershelvam, M. Suleyman, C. Beattie, S. Petersen, S. Legg, V. Mnih, K. Kavukcuoglu, and D. Silver. Massively parallel methods for deep reinforcement learning. In Deep Learning Workshop, ICML, 2015.

M. Riedmiller. Neural fitted Q iteration - first experiences with a data efficient neural reinforcement learning method. In J. Gama, R. Camacho, P. Brazdil, A. Jorge, and L. Torgo, editors, Proceedings of the 16th European Conference on Machine Learning (ECML'05), pages 317–328. Springer, 2005.

B. Sallans and G. E. Hinton. Reinforcement learning with factored states and actions. The Journal of Machine Learning Research, 5:1063–1088, 2004.

A. L. Strehl, L. Li, and M. L. Littman. Reinforcement learning in finite MDPs: PAC analysis. The Journal of Machine Learning Research, 10:2413–2444, 2009.

R. S. Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44, 1988.

R. S. Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In Proceedings of the seventh international conference on machine learning, pages 216–224, 1990.

R. S. Sutton and A. G. Barto. Introduction to reinforcement learning. MIT Press, 1998.

R. S. Sutton, C. Szepesvári, and H. R. Maei. A convergent O(n) algorithm for off-policy temporal-difference learning with linear function approximation. Advances in Neural Information Processing Systems 21 (NIPS-08), 21:1609–1616, 2008.

R. S. Sutton, A. R. Mahmood, and M. White. An emphatic approach to the problem of off-policy temporal-difference learning. arXiv preprint arXiv:1503.04269, 2015.

I. Szita and A. Lorincz. The many faces of optimism: a unifying approach. In Proceedings of the 25th international conference on Machine learning, pages 1048–1055. ACM, 2008.

G. Tesauro. Temporal difference learning and td-gammon. Communications of the ACM, 38(3):58–68, 1995.

S. Thrun and A. Schwartz. Issues in using function approximation for reinforcement learning. In M. Mozer, P. Smolensky, D. Touretzky, J. Elman, and A. Weigend, editors, Proceedings of the 1993 Connectionist Models Summer School, Hillsdale, NJ, 1993. Lawrence Erlbaum.

J. N. Tsitsiklis and B. Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control, 42(5):674–690, 1997.

H. van Hasselt. Double Q-learning. Advances in Neural Information Processing Systems, 23:2613–2621, 2010.

H. van Hasselt. Insights in Reinforcement Learning. PhD thesis, Utrecht University, 2011.

C. J. C. H. Watkins. Learning from delayed rewards. PhD thesis, University of Cambridge England, 1989.

---

## 附录

**定理 1** （完整证明）

考虑一个状态 $s$，其中所有真实最优动作值相等，即 $Q^*(s, a) = V^*(s)$。设 $Q_t$ 为任意值估计，总体无偏 $\sum_a (Q_t(s, a) - V^*(s)) = 0$，但并非全部为零，使得 $\frac{1}{m}\sum_a (Q_t(s, a) - V^*(s))^2 = C$（$C > 0$，$m \geq 2$）。则 $\max_a Q_t(s, a) \geq V^*(s) + \sqrt{\frac{C}{m-1}}$，且此下界是紧的。在相同条件下，双重Q学习估计绝对误差的下界为零。

**证明.** 定义每个动作 $a$ 的误差为 $\epsilon_a = Q_t(s, a) - V^*(s)$。假设存在 $\{\epsilon_a\}$ 的某个配置使得 $\max_a \epsilon_a < \sqrt{\frac{C}{m-1}}$。设 $\{\epsilon^+_i\}$ 为大小为 $n$ 的正 $\epsilon$ 集合，$\{\epsilon^-_j\}$ 为大小为 $m - n$ 的严格负 $\epsilon$ 集合。若 $n = m$，则 $\sum_a \epsilon_a = 0 \Rightarrow \epsilon_a = 0\ \forall a$，与 $\sum_a \epsilon^2_a = mC$ 矛盾。因此 $n \leq m - 1$。则 $\sum_{i=1}^{n} \epsilon^+_i \leq n \max_i \epsilon^+_i < n\sqrt{\frac{C}{m-1}}$，由约束 $\sum_a \epsilon_a = 0$ 得 $\sum_{j=1}^{m-n} |\epsilon^-_j| < n\sqrt{\frac{C}{m-1}}$，从而 $\max_j |\epsilon^-_j| < n\sqrt{\frac{C}{m-1}}$。由 Holder 不等式：

$$\sum_{j=1}^{m-n}(\epsilon^-_j)^2 \leq \sum_{j=1}^{m-n}|\epsilon^-_j| \cdot \max_j|\epsilon^-_j| < n\sqrt{\frac{C}{m-1}} \cdot n\sqrt{\frac{C}{m-1}}$$

综合得：

$$\sum_{a=1}^{m}(\epsilon_a)^2 = \sum_{i=1}^{n}(\epsilon^+_i)^2 + \sum_{j=1}^{m-n}(\epsilon^-_j)^2 < \frac{Cn(n+1)}{m-1} \leq mC$$

这与 $\sum_{a=1}^{m} \epsilon^2_a = mC$ 矛盾，因此 $\max_a \epsilon_a \geq \sqrt{\frac{C}{m-1}}$。设 $\epsilon_a = \sqrt{\frac{C}{m-1}}$（$a = 1, \ldots, m-1$）且 $\epsilon_m = -\sqrt{(m-1)C}$ 可验证下界紧的。

对于双重Q学习，$|Q'_t(s, \arg\max_a Q_t(s, a)) - V^*(s)|$ 的唯一紧下界为零。可通过设 $Q_t(s, a_1) = V^*(s) + \sqrt{\frac{C(m-1)}{m}}$，$Q_t(s, a_i) = V^*(s) - \sqrt{\frac{C}{m(m-1)}}$（$i > 1$），且 $Q'_t(s, a_1) = V^*(s)$ 来验证。

**定理 2.** 考虑一个状态 $s$，其中所有真实最优动作值相等 $Q^*(s, a) = V^*(s)$。假设估计误差 $Q_t(s, a) - Q^*(s, a)$ 独立且在 $[-1, 1]$ 上均匀随机分布。则：

$$\mathbb{E}\left[\max_a Q_t(s, a) - V^*(s)\right] = \frac{m-1}{m+1}$$

**证明.** 定义 $\epsilon_a = Q_t(s, a) - Q^*(s, a)$，为 $[-1, 1]$ 上的均匀随机变量。$P(\max_a \epsilon_a \leq x) = \prod_{a=1}^{m} P(\epsilon_a \leq x)$。CDF 为 $P(\epsilon_a \leq x) = \frac{1+x}{2}$（$x \in (-1, 1)$），因此 $P(\max_a \epsilon_a \leq x) = \left(\frac{1+x}{2}\right)^m$。其概率密度函数为 $f_{max}(x) = \frac{m}{2}\left(\frac{1+x}{2}\right)^{m-1}$。计算期望：

$$\mathbb{E}\left[\max_a \epsilon_a\right] = \int_{-1}^{1} x f_{max}(x) dx = \left[\left(\frac{x+1}{2}\right)^m \frac{mx-1}{m+1}\right]_{-1}^{1} = \frac{m-1}{m+1}$$

---

## 附录：Atari 2600 实验细节

### 网络架构

实验中使用的卷积网络与Mnih et al. (2015) 提出的完全一致。网络输入为 84x84x4 的张量，包含最近四帧经过缩放和灰度处理的图像。第一个卷积层使用32个 8x8 滤波器（步幅4），第二层使用64个 4x4 滤波器（步幅2），最后一个卷积层使用64个 3x3 滤波器（步幅1）。之后是512个单元的全连接隐藏层。所有层之间由修正线性单元（ReLU）分隔。最后一个全连接线性层投射到网络输出，即Q值。优化器为RMSProp（动量参数0.95）。

### 超参数

所有实验中，折扣因子 $\gamma = 0.99$，学习率 $\alpha = 0.00025$。目标网络更新间隔 $\tau = 10000$ 步。训练5000万步（即2亿帧）。每100万步评估一次智能体，保留各次评估中最佳策略作为学习输出。经验回放记忆库大小为100万条元组。每4步从记忆库中采样大小为32的小批量更新网络。探索策略为 $\epsilon$-贪心策略，$\epsilon$ 在100万步内从1线性衰减到0.1。

### 补充结果表

**表 3：** 无操作评估条件下的原始分数（5分钟模拟器时间）

| 游戏 | 随机 | 人类 | DQN | Double DQN |
|---|---|---|---|---|
| Alien | 227.80 | 6875.40 | 3069.33 | 2907.30 |
| Amidar | 5.80 | 1675.80 | 739.50 | 702.10 |
| Assault | 222.40 | 1496.40 | 3358.63 | 5022.90 |
| Asterix | 210.00 | 8503.30 | 6011.67 | 15150.00 |
| Asteroids | 719.10 | 13156.70 | 1629.33 | 930.60 |
| Atlantis | 12850.00 | 29028.10 | 85950.00 | 64758.00 |
| Bank Heist | 14.20 | 734.40 | 429.67 | 728.30 |
| Battle Zone | 2360.00 | 37800.00 | 26300.00 | 25730.00 |
| Beam Rider | 363.90 | 5774.70 | 6845.93 | 7654.00 |
| Bowling | 23.10 | 154.80 | 42.40 | 70.50 |
| Boxing | 0.10 | 4.30 | 71.83 | 81.70 |
| Breakout | 1.70 | 31.80 | 401.20 | 375.00 |
| Centipede | 2090.90 | 11963.20 | 8309.40 | 4139.40 |
| Chopper Command | 811.00 | 9881.80 | 6686.67 | 4653.00 |
| Crazy Climber | 10780.50 | 35410.50 | 114103.33 | 101874.00 |
| Demon Attack | 152.10 | 3401.30 | 9711.17 | 9711.90 |
| Double Dunk | -18.60 | -15.50 | -18.07 | -6.30 |
| Enduro | 0.00 | 309.60 | 301.77 | 319.50 |
| Fishing Derby | -91.70 | 5.50 | -0.80 | 20.30 |
| Freeway | 0.00 | 29.60 | 30.30 | 31.80 |
| Frostbite | 65.20 | 4334.70 | 328.33 | 241.50 |
| Gopher | 257.60 | 2321.00 | 8520.00 | 8215.40 |
| Gravitar | 173.00 | 2672.00 | 306.67 | 170.50 |
| H.E.R.O. | 1027.00 | 25762.50 | 19950.33 | 20357.00 |
| Ice Hockey | -11.20 | 0.90 | -1.60 | -2.40 |
| James Bond | 29.00 | 406.70 | 576.67 | 438.00 |
| Kangaroo | 52.00 | 3035.00 | 6740.00 | 13651.00 |
| Krull | 1598.00 | 2394.60 | 3804.67 | 4396.70 |
| Kung-Fu Master | 258.50 | 22736.20 | 23270.00 | 29486.00 |
| Montezuma's Revenge | 0.00 | 4366.70 | 0.00 | 0.00 |
| Ms. Pacman | 307.30 | 15693.40 | 2311.00 | 3210.00 |
| Name This Game | 2292.30 | 4076.20 | 7256.67 | 6997.10 |
| Pong | -20.70 | 9.30 | 18.90 | 21.00 |
| Private Eye | 24.90 | 69571.30 | 1787.57 | 670.10 |
| Q*Bert | 163.90 | 13455.00 | 10595.83 | 14875.00 |
| River Raid | 1338.50 | 13513.30 | 8315.67 | 12015.30 |
| Road Runner | 11.50 | 7845.00 | 18256.67 | 48377.00 |
| Robotank | 2.20 | 11.90 | 51.57 | 46.70 |
| Seaquest | 68.40 | 20181.80 | 5286.00 | 7995.00 |
| Space Invaders | 148.00 | 1652.30 | 1975.50 | 3154.60 |
| Star Gunner | 664.00 | 10250.00 | 57996.67 | 65188.00 |
| Tennis | -23.80 | -8.90 | -2.47 | 1.70 |
| Time Pilot | 3568.00 | 5925.00 | 5946.67 | 7964.00 |
| Tutankham | 11.40 | 167.60 | 186.70 | 190.60 |
| Up and Down | 533.40 | 9082.00 | 8456.33 | 16769.90 |
| Venture | 0.00 | 1187.50 | 380.00 | 93.00 |
| Video Pinball | 16256.90 | 17297.60 | 42684.07 | 70009.00 |
| Wizard of Wor | 563.50 | 4756.50 | 3393.33 | 5204.00 |
| Zaxxon | 32.50 | 9173.30 | 4976.67 | 10182.00 |

**表 5：** 人类起始点条件下的原始分数（30分钟模拟器时间）

| 游戏 | 随机 | 人类 | DQN | Double DQN | Double DQN（调优） |
|---|---|---|---|---|---|
| Alien | 128.30 | 6371.30 | 570.2 | 621.6 | 1033.4 |
| Amidar | 11.80 | 1540.40 | 133.4 | 188.2 | 169.1 |
| Assault | 166.90 | 628.90 | 3332.3 | 2774.3 | 6060.8 |
| Asterix | 164.50 | 7536.00 | 124.5 | 5285.0 | 16837.0 |
| Asteroids | 871.30 | 36517.30 | 697.1 | 1219.0 | 1193.2 |
| Atlantis | 13463.00 | 26575.00 | 76108.0 | 260556.0 | 319688.0 |
| Bank Heist | 21.70 | 644.50 | 176.3 | 469.8 | 886.0 |
| Battle Zone | 3560.00 | 33030.00 | 17560.0 | 25240.0 | 24740.0 |
| Beam Rider | 254.60 | 14961.00 | 8672.4 | 9107.9 | 17417.2 |
| Berzerk | 196.10 | 2237.50 | — | 635.8 | 1011.1 |
| Bowling | 35.20 | 146.50 | 41.2 | 62.3 | 69.6 |
| Boxing | -1.50 | 9.60 | 25.8 | 52.1 | 73.5 |
| Breakout | 1.60 | 27.90 | 303.9 | 338.7 | 368.9 |
| Centipede | 1925.50 | 10321.90 | 3773.1 | 5166.6 | 3853.5 |
| Chopper Command | 644.00 | 8930.00 | 3046.0 | 2483.0 | 3495.0 |
| Crazy Climber | 9337.00 | 32667.00 | 50992.0 | 94315.0 | 113782.0 |
| Defender | 1965.50 | 14296.00 | — | 8531.0 | 27510.0 |
| Demon Attack | 208.30 | 3442.80 | 12835.2 | 13943.5 | 69803.4 |
| Double Dunk | -16.00 | -14.40 | -21.6 | -6.4 | -0.3 |
| Enduro | -81.80 | 740.20 | 475.6 | 475.9 | 1216.6 |
| Fishing Derby | -77.10 | 5.10 | -2.3 | -3.4 | 3.2 |
| Freeway | 0.10 | 25.60 | 25.8 | 26.3 | 28.8 |
| Frostbite | 66.40 | 4202.80 | 157.4 | 258.3 | 1448.1 |
| Gopher | 250.00 | 2311.00 | 2731.8 | 8742.8 | 15253.0 |
| Gravitar | 245.50 | 3116.00 | 216.5 | 170.0 | 200.5 |
| H.E.R.O. | 1580.30 | 25839.40 | 12952.5 | 15341.4 | 14892.5 |
| Ice Hockey | -9.70 | 0.50 | -3.8 | -3.6 | -2.5 |
| James Bond | 33.50 | 368.50 | 348.5 | 416.0 | 573.0 |
| Kangaroo | 100.00 | 2739.00 | 2696.0 | 6138.0 | 11204.0 |
| Krull | 1151.90 | 2109.10 | 3864.0 | 6130.4 | 6796.1 |
| Kung-Fu Master | 304.00 | 20786.80 | 11875.0 | 22771.0 | 30207.0 |
| Montezuma's Revenge | 25.00 | 4182.00 | 50.0 | 30.0 | 42.0 |
| Ms. Pacman | 197.80 | 15375.00 | 763.5 | 1401.8 | 1241.3 |
| Name This Game | 1747.80 | 6796.00 | 5439.9 | 7871.5 | 8960.3 |
| Phoenix | 1134.40 | 6686.20 | — | 10364.0 | 12366.5 |
| Pitfall | -348.80 | 5998.90 | — | -432.9 | -186.7 |
| Pong | -18.00 | 15.50 | 16.2 | 17.7 | 19.1 |
| Private Eye | 662.80 | 64169.10 | 298.2 | 346.3 | -575.5 |
| Q*Bert | 183.00 | 12085.00 | 4589.8 | 10713.3 | 11020.8 |
| River Raid | 588.30 | 14382.20 | 4065.3 | 6579.0 | 10838.4 |
| Road Runner | 200.00 | 6878.00 | 9264.0 | 43884.0 | 43156.0 |
| Robotank | 2.40 | 8.90 | 58.5 | 52.0 | 59.1 |
| Seaquest | 215.50 | 40425.80 | 2793.9 | 4199.4 | 14498.0 |
| Skiing | -15287.40 | -3686.60 | — | -29404.3 | -11490.4 |
| Solaris | 2047.20 | 11032.60 | — | 2166.8 | 810.0 |
| Space Invaders | 182.60 | 1464.90 | 1449.7 | 1495.7 | 2628.7 |
| Star Gunner | 697.00 | 9528.00 | 34081.0 | 53052.0 | 58365.0 |
| Surround | -9.70 | 5.40 | — | -7.6 | 1.9 |
| Tennis | -21.40 | -6.70 | -2.3 | 11.0 | -7.8 |
| Time Pilot | 3273.00 | 5650.00 | 5640.0 | 5375.0 | 6608.0 |
| Tutankham | 12.70 | 138.30 | 32.4 | 63.6 | 92.2 |
| Up and Down | 707.20 | 9896.10 | 3311.3 | 4721.1 | 19086.9 |
| Venture | 18.00 | 1039.00 | 54.0 | 75.0 | 21.0 |
| Video Pinball | 20452.0 | 15641.10 | 20228.1 | 148883.6 | 367823.7 |
| Wizard of Wor | 804.00 | 4556.00 | 246.0 | 155.0 | 6201.0 |
| Yars Revenge | 1476.90 | 47135.20 | — | 5439.5 | 6270.6 |
| Zaxxon | 475.00 | 8443.00 | 831.0 | 7874.0 | 8593.0 |
