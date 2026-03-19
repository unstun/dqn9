# Rainbow：组合深度强化学习中的多项改进

**作者：** Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver

**机构：** DeepMind

**来源：** arXiv:1710.02298v1 [cs.AI], 2017年10月6日；AAAI 2018

---

## 摘要

深度强化学习（Deep Reinforcement Learning）社区已对 DQN 算法做出了多项独立改进。然而，这些扩展之间是否互补、能否有效组合仍不清楚。本文考察了 DQN 算法的六种扩展，并通过实验研究其组合效果。实验表明，该组合在 Atari 2600 基准测试中达到了当时的最优性能（state-of-the-art），无论是在数据效率（data efficiency）还是最终性能方面均表现出色。此外，我们还提供了详细的消融研究（ablation study）结果，展示了各组件对整体性能的贡献。

---

## 1 引言

将强化学习（Reinforcement Learning, RL）扩展到复杂序列决策问题的诸多成功始于深度 Q 网络算法（Deep Q-Networks, DQN；Mnih et al. 2013, 2015）。DQN 将 Q-learning 与卷积神经网络（Convolutional Neural Networks）和经验回放（Experience Replay）相结合，使智能体能够从原始像素学习玩许多 Atari 游戏，并达到人类水平。此后，许多扩展被提出以提升其速度或稳定性：

- **双重 DQN**（Double DQN, DDQN；van Hasselt, Guez, and Silver 2016）通过解耦自举动作的选择与评估，解决了 Q-learning 的过估计偏差（overestimation bias）（van Hasselt 2010）。
- **优先经验回放**（Prioritized Experience Replay；Schaul et al. 2015）通过更频繁地回放那些有更多学习价值的转移来提高数据效率。
- **竞争网络架构**（Dueling Network Architecture；Wang et al. 2016）通过分别表示状态价值（state value）和动作优势（action advantage），有助于跨动作的泛化。
- **多步自举目标**（Multi-step Bootstrap Targets；Sutton 1988; Sutton and Barto 1998）用于 A3C（Mnih et al. 2016），调节偏差-方差权衡（bias-variance trade-off），帮助将新观察到的奖励更快传播到之前访问的状态。
- **分布式 Q-learning**（Distributional Q-learning；Bellemare, Dabney, and Munos 2017）学习折扣回报（discounted return）的类别分布，而非仅估计均值。
- **噪声 DQN**（Noisy DQN；Fortunato et al. 2017）使用随机网络层进行探索。

这些算法各自独立地带来了显著的性能提升。由于它们解决的是根本不同的问题，且构建在共同的框架之上，因此有望组合在一起。已有部分组合的先例：优先 DDQN 和竞争 DDQN 都使用了双重 Q-learning，竞争 DDQN 也曾与优先经验回放结合。本文提出研究一个将上述所有成分整合在一起的智能体，并展示了这些不同的思想如何集成，以及它们确实在很大程度上互补。事实上，其组合在 Arcade Learning Environment（Bellemare et al. 2013）的 57 款 Atari 2600 游戏基准测试中取得了新的最优结果，无论是在数据效率还是最终性能方面。最后，我们展示了消融研究的结果，以帮助理解各组件的贡献。

> **图 1**：57 款 Atari 游戏上人类归一化分数的中位数。将集成智能体（Rainbow，彩虹色）与 DQN（灰色）及六种已发表的基线进行比较。Rainbow 在 7M 帧时即匹配 DQN 的最佳性能，在 44M 帧内超过所有基线，并达到大幅提升的最终性能。曲线经 5 点移动平均平滑处理。

---

## 2 背景

强化学习处理的问题是：智能体（agent）学习在环境（environment）中采取行动以最大化标量奖励信号。智能体不会获得直接监督，例如不会被直接告知最佳动作。

### 2.1 智能体与环境

在每个离散时间步 $t = 0, 1, 2, \ldots$，环境向智能体提供观测 $S_t$，智能体选择动作 $A_t$，然后环境提供下一个奖励 $R_{t+1}$、折扣因子 $\gamma_{t+1}$ 和状态 $S_{t+1}$。这种交互被形式化为马尔可夫决策过程（Markov Decision Process, MDP），即元组 $\langle \mathcal{S}, \mathcal{A}, T, r, \gamma \rangle$，其中 $\mathcal{S}$ 是有限状态集，$\mathcal{A}$ 是有限动作集，$T(s, a, s') = P[S_{t+1} = s' | S_t = s, A_t = a]$ 是（随机）转移函数，$r(s, a) = E[R_{t+1} | S_t = s, A_t = a]$ 是奖励函数，$\gamma \in [0, 1]$ 是折扣因子。在实验中 MDP 为幕式（episodic）的，恒定 $\gamma_t = \gamma$，但在幕终止时 $\gamma_t = 0$。

在智能体侧，动作选择由策略 $\pi$ 给出，它为每个状态定义动作的概率分布。从时间步 $t$ 遇到的状态 $S_t$ 出发，定义折扣回报为：

$$G_t = \sum_{k=0}^{\infty} \gamma_t^{(k)} R_{t+k+1}$$

其中 $k$ 步后奖励的折扣为 $\gamma_t^{(k)} = \prod_{i=1}^{k} \gamma_{t+i}$。智能体的目标是通过寻找好的策略来最大化期望折扣回报。

策略可以直接学习，也可以构建为其他学习量的函数。在基于价值的强化学习（value-based RL）中，智能体学习期望折扣回报的估计，即从给定状态出发遵循策略 $\pi$ 的价值：$v_\pi(s) = E_\pi[G_t | S_t = s]$，或状态-动作对的价值：$q_\pi(s, a) = E_\pi[G_t | S_t = s, A_t = a]$。从状态-动作价值函数中导出新策略的常用方式是相对于动作价值进行 $\epsilon$-贪心（$\epsilon$-greedy）行动，即以概率 $(1 - \epsilon)$ 选择最高价值的动作（贪心动作），以概率 $\epsilon$ 均匀随机选择。这类策略引入了一种探索形式，但其主要局限在于难以发现延伸到远期未来的替代行动方案，这促进了更有方向性的探索方法的研究。

### 2.2 深度强化学习与 DQN

大的状态和/或动作空间使得为每个状态-动作对独立学习 Q 值估计变得不可行。在深度强化学习中，用深层（多层）神经网络来表示智能体的各种组件，如策略 $\pi(s, a)$ 或价值 $q(s, a)$，网络参数通过梯度下降最小化合适的损失函数来训练。

在 DQN（Mnih et al. 2015）中，深度网络与强化学习成功结合：使用卷积神经网络近似给定状态 $S_t$（以原始像素帧堆叠作为网络输入）的动作价值。每一步，智能体基于当前状态相对于动作价值进行 $\epsilon$-贪心选择动作，并将转移 $(S_t, A_t, R_{t+1}, \gamma_{t+1}, S_{t+1})$ 添加到经验回放缓冲区（replay memory buffer）（Lin 1992），该缓冲区保存最近一百万个转移。网络参数通过随机梯度下降最小化以下损失来优化：

$$(R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a') - q_\theta(S_t, A_t))^2 \tag{1}$$

其中 $t$ 是从回放缓冲区中随机选取的时间步。损失梯度仅反向传播到在线网络（online network）的参数 $\theta$（该网络也用于选择动作）；$\bar{\theta}$ 表示目标网络（target network）的参数，它是在线网络的周期性副本，不直接优化。优化使用 RMSprop（Tieleman and Hinton 2012），从经验回放中均匀采样的小批量上执行。经验回放和目标网络的使用实现了相对稳定的 Q 值学习，并在多款 Atari 游戏上达到了超人类水平。

---

## 3 DQN 的扩展

DQN 是一个重要里程碑，但该算法的几个局限性现已为人所知，且已提出许多扩展。本文选取了六种扩展，每种都解决了一个不同的局限性并提升了整体性能。

### 3.1 双重 Q-learning（Double Q-learning）

传统 Q-learning 由于公式 (1) 中的最大化步骤而受到过估计偏差的影响，这可能损害学习。双重 Q-learning（van Hasselt 2010）通过在自举目标的最大化操作中解耦动作的选择和评估来解决此问题。将其与 DQN 结合（van Hasselt, Guez, and Silver 2016），使用以下损失：

$$(R_{t+1} + \gamma_{t+1} q_{\bar{\theta}}(S_{t+1}, \arg\max_{a'} q_\theta(S_{t+1}, a')) - q_\theta(S_t, A_t))^2$$

这一改变被证明减少了 DQN 中存在的有害过估计，从而提升了性能。

### 3.2 优先回放（Prioritized Replay）

DQN 从回放缓冲区中均匀采样。理想情况下，我们希望更频繁地采样那些有更多学习价值的转移。作为学习潜力的代理指标，优先经验回放（Schaul et al. 2015）以概率 $p_t$ 采样转移，该概率与最近遇到的绝对 TD 误差成比例：

$$p_t \propto |R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a') - q_\theta(S_t, A_t)|^\omega$$

其中 $\omega$ 是决定分布形状的超参数。新转移以最大优先级插入回放缓冲区，提供了对近期转移的偏向。注意，随机转移也可能被偏好，即使从它们身上已经没有太多可学。

### 3.3 竞争网络（Dueling Networks）

竞争网络是一种为基于价值的 RL 设计的神经网络架构。它包含两个计算流——价值流（value stream）和优势流（advantage stream），共享一个卷积编码器，并通过一个特殊的聚合器合并（Wang et al. 2016）。这对应于以下动作价值分解：

$$q_\theta(s, a) = v_\eta(f_\xi(s)) + a_\psi(f_\xi(s), a) - \frac{\sum_{a'} a_\psi(f_\xi(s), a')}{N_{\text{actions}}}$$

其中 $\xi$、$\eta$ 和 $\psi$ 分别是共享编码器 $f_\xi$、价值流 $v_\eta$ 和优势流 $a_\psi$ 的参数；$\theta = \{\xi, \eta, \psi\}$ 是它们的连接。

### 3.4 多步学习（Multi-step Learning）

Q-learning 累积单步奖励然后用下一步的贪心动作进行自举。作为替代方案，可以使用前视多步目标（Sutton 1988）。定义从给定状态 $S_t$ 出发的截断 $n$ 步回报为：

$$R_t^{(n)} \equiv \sum_{k=0}^{n-1} \gamma_t^{(k)} R_{t+k+1} \tag{2}$$

DQN 的多步变体通过最小化以下替代损失来定义：

$$(R_t^{(n)} + \gamma_t^{(n)} \max_{a'} q_{\bar{\theta}}(S_{t+n}, a') - q_\theta(S_t, A_t))^2$$

适当调参的多步目标通常能带来更快的学习（Sutton and Barto 1998）。

### 3.5 分布式强化学习（Distributional RL）

我们可以学习近似回报的分布，而不仅仅是期望回报。Bellemare, Dabney, and Munos (2017) 提出将此类分布建模为放置在离散支撑 $z$ 上的概率质量，其中 $z$ 是具有 $N_{\text{atoms}} \in \mathbb{N}^+$ 个原子的向量，定义为：

$$z_i = v_{\min} + (i - 1) \frac{v_{\max} - v_{\min}}{N_{\text{atoms}} - 1}, \quad i \in \{1, \ldots, N_{\text{atoms}}\}$$

时间步 $t$ 的近似分布 $d_t$ 定义在该支撑上，每个原子 $i$ 上有概率质量 $p_\theta^i(S_t, A_t)$，使得 $d_t = (z, p_\theta(S_t, A_t))$。目标是更新 $\theta$ 使该分布紧密匹配回报的实际分布。

回报分布满足 Bellman 方程的一个变体。对于给定状态 $S_t$ 和动作 $A_t$，最优策略 $\pi^*$ 下的回报分布应匹配一个目标分布，该目标分布通过取下一个状态 $S_{t+1}$ 和动作 $a_{t+1}^* = \pi^*(S_{t+1})$ 的分布，按折扣收缩并按奖励平移得到。分布式 Q-learning 的变体通过首先为目标分布构建新的支撑，然后最小化分布 $d_t$ 与目标分布之间的 KL 散度（Kullback-Leibler divergence）来推导：

$$d_t' \equiv (R_{t+1} + \gamma_{t+1} z, \; p_{\bar{\theta}}(S_{t+1}, a_{t+1}^*))$$

$$D_{\text{KL}}(\Phi_z d_t' \| d_t) \tag{3}$$

其中 $\Phi_z$ 是目标分布到固定支撑 $z$ 的 $L_2$ 投影，$a_{t+1}^* = \arg\max_a q_{\bar{\theta}}(S_{t+1}, a)$ 是关于状态 $S_{t+1}$ 中均值动作价值 $q_{\bar{\theta}}(S_{t+1}, a) = z^\top p_{\bar{\theta}}(S_{t+1}, a)$ 的贪心动作。

参数化分布可由神经网络表示，类似 DQN，但具有 $N_{\text{atoms}} \times N_{\text{actions}}$ 个输出。对输出的每个动作维度独立应用 softmax，以确保每个动作的分布被适当归一化。

### 3.6 噪声网络（Noisy Nets）

使用 $\epsilon$-贪心策略进行探索的局限性在如 Montezuma's Revenge 等游戏中很明显，在这些游戏中必须执行许多动作才能收集到第一个奖励。Noisy Nets（Fortunato et al. 2017）提出了一种噪声线性层，结合确定性流和噪声流：

$$y = (b + Wx) + (b_{\text{noisy}} \odot \epsilon_b + (W_{\text{noisy}} \odot \epsilon_w)x) \tag{4}$$

其中 $\epsilon_b$ 和 $\epsilon_w$ 是随机变量，$\odot$ 表示逐元素乘积。这种变换可以替代标准线性变换 $y = b + Wx$。随着时间推移，网络可以学习忽略噪声流，但在状态空间的不同部分以不同速率进行，从而实现状态条件探索和一种自退火（self-annealing）形式。

---

## 4 集成智能体

本文将上述所有组件整合为一个集成智能体，称为 **Rainbow**。

首先，用多步变体替代单步分布式损失 (3)。通过按累积折扣收缩 $S_{t+n}$ 处的价值分布，并按截断 $n$ 步折扣回报平移来构建目标分布：

$$d_t^{(n)} = (R_t^{(n)} + \gamma_t^{(n)} z, \; p_{\bar{\theta}}(S_{t+n}, a_{t+n}^*))$$

损失为：

$$D_{\text{KL}}(\Phi_z d_t^{(n)} \| d_t)$$

其中 $\Phi_z$ 是到 $z$ 的投影。

通过使用在线网络在 $S_{t+n}$ 中选择的贪心动作作为自举动作 $a_{t+n}^*$，并用目标网络评估该动作，将多步分布式损失与双重 Q-learning 结合。

在标准比例优先回放（Schaul et al. 2015）中使用绝对 TD 误差来优先化转移。在分布式设置中，所有分布式 Rainbow 变体通过 KL 损失来优先化转移，因为这正是算法最小化的目标：

$$p_t \propto (D_{\text{KL}}(\Phi_z d_t^{(n)} \| d_t))^\omega$$

KL 损失作为优先级可能对噪声随机环境更鲁棒，因为即使回报不是确定性的，损失也可以继续下降。

网络架构采用了适配回报分布的竞争网络架构。网络有共享表示 $f_\xi(s)$，然后输入到具有 $N_{\text{atoms}}$ 个输出的价值流 $v_\eta$ 和具有 $N_{\text{atoms}} \times N_{\text{actions}}$ 个输出的优势流 $a_\psi$。对于每个原子 $z_i$，价值流和优势流如竞争 DQN 中那样聚合，然后通过 softmax 层获得归一化参数分布，用于估计回报分布：

$$p_\theta^i(s, a) = \frac{\exp(v_\eta^i(\phi) + a_\psi^i(\phi, a) - \bar{a}_\psi^i(s))}{\sum_j \exp(v_\eta^j(\phi) + a_\psi^j(\phi, a) - \bar{a}_\psi^j(s))}$$

其中 $\phi = f_\xi(s)$，$\bar{a}_\psi^i(s) = \frac{1}{N_{\text{actions}}} \sum_{a'} a_\psi^i(\phi, a')$。

然后将所有线性层替换为公式 (4) 中描述的噪声等价层。在这些噪声线性层中使用分解高斯噪声（factorised Gaussian noise）（Fortunato et al. 2017）以减少独立噪声变量的数量。

---

## 5 实验方法

### 5.1 评估方法

在 Arcade Learning Environment（Bellemare et al. 2013）的 57 款 Atari 2600 游戏上评估所有智能体。遵循 Mnih et al. (2015) 和 van Hasselt et al. (2016) 的训练和评估流程。在训练过程中每 1M 步暂停学习，用最新的智能体评估 500K 帧来计算平均分数。每幕截断在 108K 帧（约 30 分钟模拟游戏时间）。

智能体分数按游戏归一化，使 0% 对应随机智能体，100% 对应人类专家的平均分数。归一化分数可在所有 Atari 关卡上聚合以比较不同智能体的性能。常用的指标是跨所有游戏的人类归一化性能的中位数。此外还考虑智能体性能超过人类某一比例的游戏数量，以区分中位数改进的来源。

训练结束后，在两种不同的测试模式下重新评估最佳智能体快照：
- **无操作启动模式**（no-ops starts）：在每幕开头插入随机数量（最多 30 个）的空操作。
- **人类启动模式**（human starts）：用从人类专家轨迹初始部分随机采样的点来初始化各幕（Nair et al. 2015）；两种模式之间的差异表明智能体对其自身轨迹过拟合的程度。

### 5.2 超参数调优

所有 Rainbow 组件都有一些超参数。超参数的组合空间太大，无法进行穷举搜索，因此只进行了有限的调优。对于每个组件，从引入该组件的论文中使用的值开始，通过手动坐标下降调优最敏感的超参数。

主要发现包括：
- 使用优先回放后，可以更早开始学习（仅需 80K 帧而非 200K 帧）。
- 使用 Noisy Nets 时完全贪心行动（$\epsilon = 0$），$\sigma_0 = 0.5$。不使用 Noisy Nets 时，将 $\epsilon$ 在前 250K 帧内退火到 0.01。
- 使用 Adam 优化器（Kingma and Ba 2014），学习率 $\alpha/4 = 0.0000625$，Adam 的 $\epsilon$ 参数为 $1.5 \times 10^{-4}$。
- 优先回放使用比例变体，优先级指数 $\omega = 0.5$，重要性采样指数 $\beta$ 从 0.4 线性增加到 1。
- 多步学习 $n = 3$ 表现最佳。

> **表 1：Rainbow 超参数**
>
> | 参数 | 值 |
> |------|------|
> | 开始学习所需最小历史 | 80K 帧 |
> | Adam 学习率 | 0.0000625 |
> | 探索 $\epsilon$ | 0.0 |
> | Noisy Nets $\sigma_0$ | 0.5 |
> | 目标网络更新周期 | 32K 帧 |
> | Adam $\epsilon$ | $1.5 \times 10^{-4}$ |
> | 优先化类型 | 比例（proportional） |
> | 优先化指数 $\omega$ | 0.5 |
> | 优先化重要性采样 $\beta$ | $0.4 \to 1.0$ |
> | 多步回报 $n$ | 3 |
> | 分布原子数 | 51 |
> | 分布最小/最大值 | $[-10, 10]$ |

---

## 6 分析

### 6.1 与已发表基线的比较

> **图 2**：每个子图显示多个智能体在达到人类性能特定比例的游戏数量随时间的变化。从左到右分别考虑 20%、50%、100%、200% 和 500% 阈值。第一行将 Rainbow 与基线比较，第二行将 Rainbow 与其消融变体比较。

Rainbow 的性能显著优于所有基线，无论是在数据效率还是最终性能方面。在 7M 帧时即匹配 DQN 的最终性能，在 44M 帧内超过这些基线的最佳最终性能，并达到大幅提升的最终性能。

在最终评估中，Rainbow 在无操作启动模式下达到了 223% 的中位数分数，在人类启动模式下达到了 153% 的中位数分数。

> **表 2：Rainbow 与基线的最佳智能体快照中位数归一化分数**
>
> | 智能体 | 无操作启动 | 人类启动 |
> |--------|-----------|----------|
> | DQN | 79% | 68% |
> | DDQN (*) | 117% | 110% |
> | 优先 DDQN (*) | 140% | 128% |
> | 竞争 DDQN (*) | 151% | 117% |
> | A3C (*) | - | 116% |
> | Noisy DQN | 118% | 102% |
> | 分布式 DQN | 164% | 125% |
> | **Rainbow** | **223%** | **153%** |
>
> 标有星号 (*) 的方法分数来自对应论文。

Rainbow 与其他智能体之间的性能差距在所有性能水平上都很明显：Rainbow 智能体在基线智能体已经表现良好的游戏上进一步提升了分数，同时在基线智能体仍远低于人类性能的游戏上也有所改进。

### 6.2 学习速度

每个智能体在单个 GPU 上运行。匹配 DQN 最终性能所需的 7M 帧对应不到 10 小时的实际时间。完整的 200M 帧运行大约需要 10 天，所有讨论的变体之间差异不超过 20%。

### 6.3 消融研究

> **图 3**：57 款 Atari 游戏上人类归一化性能的中位数随时间变化。将集成智能体（Rainbow，彩虹色）与 DQN（灰色）和六种不同的消融变体（虚线）进行比较。曲线经 5 点移动平均平滑处理。

> **图 4**：消融智能体在所有 57 款 Atari 游戏上的性能下降。性能为学习曲线下面积，相对于 Rainbow 智能体和 DQN 归一化。省略了 DQN 优于 Rainbow 的两款游戏。为每款游戏突出显示了导致最大下降的消融。移除优先化或多步学习在大多数游戏中降低了性能，但每个组件的贡献在不同游戏中差异很大。

在每次消融中，从完整 Rainbow 组合中移除一个组件。主要发现如下：

1. **优先回放和多步学习**是 Rainbow 中最关键的两个组件，移除任一组件都会导致中位数性能的大幅下降。两者不仅影响早期性能，多步学习的移除还损害了最终性能。在逐游戏分析中，两个组件在 57 款游戏中的 53 款上帮助了 Rainbow。

2. **分布式 Q-learning** 在相关性方面紧随其后。值得注意的是，在早期学习中没有明显差异——前 4000 万帧中，消融了分布式的变体与完整智能体表现一样好。但之后，没有分布的智能体性能开始落后。分布式消融主要在接近或超过人类水平的游戏上表现出差距。

3. **Noisy Nets**：总体而言，包含 Noisy Nets 时智能体表现更好；移除后探索退回到传统 $\epsilon$-贪心机制，聚合性能更差。虽然移除 Noisy Nets 在多款游戏上导致大幅性能下降，但在其他游戏上也带来了小幅提升。

4. **竞争网络**：在聚合层面，移除竞争网络后未观察到显著差异。中位数分数掩盖了竞争网络在不同游戏间影响不同的事实：在超人类性能水平的游戏上可能提供了一些改进，在低于人类性能的游戏上则有一些退化。

5. **双重 Q-learning**：在中位数性能上差异有限，视游戏不同有时有害有时有益。进一步调查发现，实际回报通常高于 10，超出了分布的支撑范围 $[-10, +10]$，导致的是回报低估而非过估计。作者推测将值裁剪到此受限范围抵消了 Q-learning 的过估计偏差。然而，如果扩展分布的支撑范围，双重 Q-learning 的重要性可能会增加。

---

## 7 讨论

本文证明了对 DQN 的多项改进可以成功整合到一个单一学习算法中，达到最优性能。此外，在集成算法中，除一项外所有组件都提供了明确的性能收益。

还有许多算法组件未能纳入，它们是进一步集成智能体实验的有前途的候选者。作者讨论了以下几个方向：

- **策略优化方法**：未考虑纯基于策略的 RL 算法，如信任域策略优化（Trust Region Policy Optimization；Schulman et al. 2015），以及演员-评论家方法（Actor-Critic；Mnih et al. 2016; O'Donoghue et al. 2016）。

- **序列数据利用**：最优性收紧（Optimality Tightening；He et al. 2016）使用多步回报构建额外的不等式约束。资格迹（Eligibility Traces）允许对 $n$ 步回报进行软组合（Sutton 1988）。但序列方法比 Rainbow 使用的多步目标需要更多计算。

- **情景控制**（Episodic Control；Blundell et al. 2016）：通过使用情景记忆作为互补学习系统，专注于数据效率，在某些领域非常有效。

- **替代探索方法**：自举 DQN（Bootstrapped DQN；Osband et al. 2016）、内在动机（Intrinsic Motivation；Stadie, Levine, and Abbeel 2015）和基于计数的探索（Count-based Exploration；Bellemare et al. 2016）。

- **并行计算架构**：如 A3C（Mnih et al. 2016）、Gorila（Nair et al. 2015）、进化策略（Evolution Strategies；Salimans et al. 2017）中的异步学习。

- **分层强化学习**（Hierarchical RL）：如 h-DQN（Kulkarni et al. 2016a）和封建网络（Feudal Networks；Vezhnevets et al. 2017）。

- **辅助任务**：像素控制或特征控制（Jaderberg et al. 2016）、监督预测（Dosovitskiy and Koltun 2016）、后继特征（Successor Features；Kulkarni et al. 2016b）。

- **领域修改改进**：Pop-Art 归一化（van Hasselt et al. 2016）允许移除奖励裁剪；细粒度动作重复（Sharma, Lakshminarayanan, and Ravindran 2017）学习如何重复动作；循环状态网络（Hausknecht and Stone 2015）可学习时序状态表示以替代固定帧堆叠。

---

## 附录

### 预处理超参数

> **表 3：预处理参数（与 DQN 及其变体相同）**
>
> | 超参数 | 值 |
> |--------|------|
> | 灰度化 | True |
> | 观测降采样 | (84, 84) |
> | 帧堆叠数 | 4 |
> | 动作重复 | 4 |
> | 奖励裁剪 | [-1, 1] |
> | 失去生命时终止 | True |
> | 每幕最大帧数 | 108K |

### 额外超参数

> **表 4：额外超参数（与 DQN 及其变体相同）**
>
> | 超参数 | 值 |
> |--------|------|
> | Q 网络：通道数 | 32, 64, 64 |
> | Q 网络：卷积核大小 | 8×8, 4×4, 3×3 |
> | Q 网络：步幅 | 4, 2, 1 |
> | Q 网络：隐藏单元 | 512 |
> | Q 网络：输出单元 | 动作数量 |
> | 折扣因子 | 0.99 |
> | 记忆大小 | 1M 转移 |
> | 回放周期 | 每 4 个智能体步 |
> | 小批量大小 | 32 |

### 详细分数表

附录中包含两个完整的分数表：

- **表 5**（人类启动评估模式）和 **表 6**（无操作启动评估模式）：列出了 57 款游戏中 Rainbow 及各基线方法的原始分数，取自 200 个测试幕的平均值。Rainbow 在大多数游戏中均取得了最高或接近最高的分数。

### 学习曲线

- **图 5**：Rainbow 与基线方法在每款游戏上的学习曲线，经 10 点移动平均平滑。
- **图 6**：Rainbow 与其消融变体在每款游戏上的学习曲线，经 10 点移动平均平滑。

---

## 参考文献

Bellemare, M. G.; Naddaf, Y.; Veness, J.; and Bowling, M. 2013. The arcade learning environment: An evaluation platform for general agents. J. Artif. Intell. Res. (JAIR) 47:253–279.

Bellemare, M. G.; Srinivasan, S.; Ostrovski, G.; Schaul, T.; Saxton, D.; and Munos, R. 2016. Unifying count-based exploration and intrinsic motivation. In NIPS.

Bellemare, M. G.; Dabney, W.; and Munos, R. 2017. A distributional perspective on reinforcement learning. In ICML.

Blundell, C.; Uria, B.; Pritzel, A.; Li, Y.; Ruderman, A.; Leibo, J. Z.; Rae, J.; Wierstra, D.; and Hassabis, D. 2016. Model-Free Episodic Control. ArXiv e-prints.

Dosovitskiy, A., and Koltun, V. 2016. Learning to act by predicting the future. CoRR abs/1611.01779.

Fortunato, M.; Azar, M. G.; Piot, B.; Menick, J.; Osband, I.; Graves, A.; Mnih, V.; Munos, R.; Hassabis, D.; Pietquin, O.; Blundell, C.; and Legg, S. 2017. Noisy networks for exploration. CoRR abs/1706.10295.

Hausknecht, M., and Stone, P. 2015. Deep recurrent Q-learning for partially observable MDPs. arXiv preprint arXiv:1507.06527.

He, F. S.; Liu, Y.; Schwing, A. G.; and Peng, J. 2016. Learning to play in a day: Faster deep reinforcement learning by optimality tightening. CoRR abs/1611.01606.

Jaderberg, M.; Mnih, V.; Czarnecki, W. M.; Schaul, T.; Leibo, J. Z.; Silver, D.; and Kavukcuoglu, K. 2016. Reinforcement learning with unsupervised auxiliary tasks. CoRR abs/1611.05397.

Kingma, D. P., and Ba, J. 2014. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR).

Kulkarni, T. D.; Narasimhan, K.; Saeedi, A.; and Tenenbaum, J. B. 2016a. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. CoRR abs/1604.06057.

Kulkarni, T. D.; Saeedi, A.; Gautam, S.; and Gershman, S. J. 2016b. Deep successor reinforcement learning. arXiv preprint arXiv:1606.02396.

Lin, L.-J. 1992. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning 8(3):293–321.

Mnih, V.; Kavukcuoglu, K.; Silver, D.; Graves, A.; Antonoglou, I.; Wierstra, D.; and Riedmiller, M. A. 2013. Playing atari with deep reinforcement learning. CoRR abs/1312.5602.

Mnih, V.; Kavukcuoglu, K.; Silver, D.; Rusu, A. A.; Veness, J.; Bellemare, M. G.; Graves, A.; Riedmiller, M.; Fidjeland, A. K.; Ostrovski, G.; Petersen, S.; Beattie, C.; Sadik, A.; Antonoglou, I.; King, H.; Kumaran, D.; Wierstra, D.; Legg, S.; and Hassabis, D. 2015. Human-level control through deep reinforcement learning. Nature 518(7540):529–533.

Mnih, V.; Badia, A. P.; Mirza, M.; Graves, A.; Lillicrap, T.; Harley, T.; Silver, D.; and Kavukcuoglu, K. 2016. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning.

Nair, A.; Srinivasan, P.; Blackwell, S.; Alcicek, C.; Fearon, R.; De Maria, A.; Panneershelvam, V.; Suleyman, M.; Beattie, C.; Petersen, S.; Legg, S.; Mnih, V.; Kavukcuoglu, K.; and Silver, D. 2015. Massively parallel methods for deep reinforcement learning. arXiv preprint arXiv:1507.04296.

O'Donoghue, B.; Munos, R.; Kavukcuoglu, K.; and Mnih, V. 2016. Pgq: Combining policy gradient and q-learning. CoRR abs/1611.01626.

Osband, I.; Blundell, C.; Pritzel, A.; and Roy, B. V. 2016. Deep exploration via bootstrapped dqn. In NIPS.

Salimans, T.; Ho, J.; Chen, X.; and Sutskever, I. 2017. Evolution strategies as a scalable alternative to reinforcement learning. CoRR abs/1703.03864.

Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015. Prioritized experience replay. In Proc. of ICLR.

Schulman, J.; Levine, S.; Moritz, P.; Jordan, M.; and Abbeel, P. 2015. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning - Volume 37, ICML'15, 1889–1897. JMLR.org.

Sharma, S.; Lakshminarayanan, A. S.; and Ravindran, B. 2017. Learning to repeat: Fine grained action repetition for deep reinforcement learning. arXiv preprint arXiv:1702.06054.

Stadie, B. C.; Levine, S.; and Abbeel, P. 2015. Incentivizing exploration in reinforcement learning with deep predictive models. CoRR abs/1507.00814.

Sutton, R. S., and Barto, A. G. 1998. Reinforcement Learning: An Introduction. The MIT press, Cambridge MA.

Sutton, R. S. 1988. Learning to predict by the methods of temporal differences. Machine learning 3(1):9–44.

Tieleman, T., and Hinton, G. 2012. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning 4(2):26–31.

van Hasselt, H.; Guez, A.; Guez, A.; Hessel, M.; Mnih, V.; and Silver, D. 2016. Learning values across many orders of magnitude. In Advances in Neural Information Processing Systems 29, 4287–4295.

van Hasselt, H.; Guez, A.; and Silver, D. 2016. Deep reinforcement learning with double Q-learning. In Proc. of AAAI, 2094–2100.

van Hasselt, H. 2010. Double Q-learning. In Advances in Neural Information Processing Systems 23, 2613–2621.

Vezhnevets, A. S.; Osindero, S.; Schaul, T.; Heess, N.; Jaderberg, M.; Silver, D.; and Kavukcuoglu, K. 2017. Feudal networks for hierarchical reinforcement learning. CoRR abs/1703.01161.

Wang, Z.; Schaul, T.; Hessel, M.; van Hasselt, H.; Lanctot, M.; and de Freitas, N. 2016. Dueling network architectures for deep reinforcement learning. In Proceedings of The 33rd International Conference on Machine Learning, 1995–2003.
