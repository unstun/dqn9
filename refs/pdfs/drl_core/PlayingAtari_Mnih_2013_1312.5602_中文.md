# 用深度强化学习玩 Atari 游戏

**作者：** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller

**机构：** DeepMind Technologies

**原文：** arXiv:1312.5602v1 [cs.LG] 2013年12月19日

---

## 摘要

我们提出了首个利用强化学习（Reinforcement Learning）从高维感知输入中直接成功学习控制策略的深度学习模型。该模型是一个卷积神经网络（Convolutional Neural Network, CNN），使用 Q-learning 的一种变体进行训练，其输入为原始像素，输出为估计未来奖励的价值函数（Value Function）。我们将该方法应用于来自街机学习环境（Arcade Learning Environment）的七个 Atari 2600 游戏，且未对网络架构或学习算法做任何调整。实验发现，该方法在其中六个游戏上超越了所有先前方法，并在三个游戏上超过了人类专家。

---

## 1 引言

直接从视觉和语音等高维感知输入中学习控制智能体，是强化学习长期以来面临的挑战之一。大多数在这些领域取得成功的强化学习应用都依赖于手工设计的特征与线性价值函数或策略表示的结合。显然，这类系统的性能严重依赖于特征表示的质量。

深度学习的最新进展使得从原始感知数据中提取高级特征成为可能，从而在计算机视觉 [11, 22, 16] 和语音识别 [6, 7] 方面取得了突破。这些方法采用了多种神经网络架构，包括卷积网络、多层感知机（Multilayer Perceptron）、受限玻尔兹曼机（Restricted Boltzmann Machine）和循环神经网络（Recurrent Neural Network），并同时利用了监督学习和无监督学习。自然而然地，人们会问类似的技术是否也能使基于感知数据的强化学习受益。

然而，从深度学习的角度来看，强化学习带来了若干挑战。首先，迄今为止大多数成功的深度学习应用需要大量手工标注的训练数据。而强化学习算法则必须从标量奖励信号中学习，该信号往往是稀疏的、含噪的且具有延迟性。动作与最终奖励之间的延迟可能长达数千个时间步，与监督学习中输入和目标之间的直接关联相比，这一点尤其具有挑战性。另一个问题是，大多数深度学习算法假设数据样本是独立的，而在强化学习中，通常会遇到高度相关的状态序列。此外，在强化学习中，数据分布会随着算法学习新行为而变化，这对假设固定底层分布的深度学习方法可能造成问题。

本文证明了卷积神经网络能够克服这些挑战，从复杂强化学习环境中的原始视频数据学习成功的控制策略。网络使用 Q-learning [26] 算法的一种变体进行训练，采用随机梯度下降（Stochastic Gradient Descent, SGD）更新权重。为了缓解相关数据和非平稳分布的问题，我们使用了经验回放（Experience Replay）机制 [13]，通过随机采样先前的转移来平滑训练分布。

我们将该方法应用于在街机学习环境（ALE）[3] 中实现的一系列 Atari 2600 游戏。Atari 2600 是一个具有挑战性的强化学习测试平台，它向智能体呈现高维视觉输入（210 $\times$ 160 RGB 视频，60Hz）以及多样且有趣的任务集。我们的目标是创建一个单一的神经网络智能体，使其能够成功学习尽可能多的游戏。网络未获得任何游戏特定信息或手工设计的视觉特征，也无法访问模拟器的内部状态；它仅从视频输入、奖励和终止信号以及可能的动作集合中学习——就像人类玩家一样。此外，网络架构和所有用于训练的超参数在不同游戏之间保持不变。到目前为止，该网络在我们尝试的七个游戏中的六个上超越了所有先前的强化学习算法，并在其中三个游戏上超过了人类专家。

> **图 1：** 五个 Atari 2600 游戏的截图（从左到右）：Pong、Breakout、Space Invaders、Seaquest、Beam Rider。

---

## 2 背景

我们考虑智能体与环境 $\mathcal{E}$（此处为 Atari 模拟器）通过一系列动作、观测和奖励进行交互的任务。在每个时间步，智能体从合法游戏动作集 $\mathcal{A} = \{1, \ldots, K\}$ 中选择一个动作 $a_t$。该动作被传递给模拟器，并修改其内部状态和游戏分数。一般而言，$\mathcal{E}$ 可能是随机的。模拟器的内部状态不被智能体观测到；相反，它从模拟器中观测到一幅图像 $x_t \in \mathbb{R}^d$，即表示当前屏幕的原始像素值向量。此外，它接收一个奖励 $r_t$，表示游戏分数的变化。注意，一般而言，游戏分数可能取决于整个先前的动作和观测序列；关于某个动作的反馈可能在数千个时间步之后才会收到。

由于智能体只能观测当前屏幕的图像，因此任务是部分可观测的（Partially Observed），且许多模拟器状态在感知上是混叠的（Perceptually Aliased），即仅从当前屏幕 $x_t$ 无法完全理解当前状况。因此我们考虑动作和观测的序列 $s_t = x_1, a_1, x_2, \ldots, a_{t-1}, x_t$，并学习依赖于这些序列的游戏策略。模拟器中的所有序列假设在有限时间步内终止。这种形式化产生了一个大但有限的马尔可夫决策过程（Markov Decision Process, MDP），其中每个序列是一个不同的状态。因此，我们可以将标准的强化学习方法应用于 MDP，只需将完整序列 $s_t$ 作为时间 $t$ 的状态表示即可。

智能体的目标是通过选择动作与模拟器交互，以最大化未来奖励。我们采用标准假设，即未来奖励按每个时间步的因子 $\gamma$ 进行折扣，并定义时间 $t$ 的未来折扣回报为 $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$，其中 $T$ 是游戏终止的时间步。我们定义最优动作价值函数（Optimal Action-Value Function）$Q^*(s, a)$ 为在观测到某个序列 $s$ 并采取某个动作 $a$ 后，通过遵循任何策略所能获得的最大期望回报：

$$Q^*(s, a) = \max_\pi \mathbb{E}[R_t | s_t = s, a_t = a, \pi]$$

其中 $\pi$ 是从序列到动作（或动作分布）的策略映射。

最优动作价值函数满足一个重要的恒等式，即贝尔曼方程（Bellman Equation）。其直觉如下：如果下一时间步序列 $s'$ 的最优值 $Q^*(s', a')$ 对于所有可能的动作 $a'$ 是已知的，则最优策略是选择使 $r + \gamma Q^*(s', a')$ 的期望值最大化的动作 $a'$：

$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{E}} \left[ r + \gamma \max_{a'} Q^*(s', a') \,\middle|\, s, a \right] \tag{1}$$

许多强化学习算法的基本思想是利用贝尔曼方程作为迭代更新来估计动作价值函数：$Q_{i+1}(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q_i(s', a') | s, a]$。这种值迭代（Value Iteration）算法收敛到最优动作价值函数，$Q_i \to Q^*$（当 $i \to \infty$）[23]。然而在实践中，这种基本方法是完全不可行的，因为动作价值函数对每个序列单独估计，没有任何泛化能力。取而代之的是使用函数逼近器（Function Approximator）来估计动作价值函数，$Q(s, a; \theta) \approx Q^*(s, a)$。在强化学习社区中，这通常是一个线性函数逼近器，但有时也使用非线性函数逼近器，如神经网络。我们将具有权重 $\theta$ 的神经网络函数逼近器称为 Q 网络（Q-Network）。Q 网络可以通过最小化一系列在每次迭代 $i$ 时变化的损失函数 $L_i(\theta_i)$ 来训练：

$$L_i(\theta_i) = \mathbb{E}_{s,a \sim \rho(\cdot)} \left[ \left( y_i - Q(s, a; \theta_i) \right)^2 \right] \tag{2}$$

其中 $y_i = \mathbb{E}_{s' \sim \mathcal{E}} [r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) | s, a]$ 是第 $i$ 次迭代的目标值，$\rho(s, a)$ 是序列 $s$ 和动作 $a$ 上的概率分布，我们称之为行为分布（Behaviour Distribution）。在优化损失函数 $L_i(\theta_i)$ 时，前一次迭代的参数 $\theta_{i-1}$ 保持固定。注意，目标值依赖于网络权重，这与监督学习中在学习开始前就固定的目标值形成了对比。对损失函数关于权重求导，我们得到以下梯度：

$$\nabla_{\theta_i} L_i(\theta_i) = \mathbb{E}_{s,a \sim \rho(\cdot); s' \sim \mathcal{E}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right) \nabla_{\theta_i} Q(s, a; \theta_i) \right] \tag{3}$$

与计算上述梯度中的完整期望值相比，通过随机梯度下降来优化损失函数在计算上通常更为高效。如果在每个时间步后更新权重，并将期望值分别替换为来自行为分布 $\rho$ 和模拟器 $\mathcal{E}$ 的单个样本，我们就得到了经典的 Q-learning 算法 [26]。

注意，该算法是无模型的（Model-Free）：它直接使用来自模拟器 $\mathcal{E}$ 的样本来解决强化学习任务，而无需显式构建 $\mathcal{E}$ 的估计。它也是离策略的（Off-Policy）：它学习贪心策略 $a = \max_a Q(s, a; \theta)$，同时遵循一个确保充分探索状态空间的行为分布。在实践中，行为分布通常通过 $\epsilon$-贪心（$\epsilon$-Greedy）策略来选择，即以概率 $1 - \epsilon$ 遵循贪心策略，以概率 $\epsilon$ 选择随机动作。

---

## 3 相关工作

也许强化学习最著名的成功案例是 TD-Gammon，这是一个完全通过强化学习和自我对弈学习的西洋双陆棋程序，达到了超人水平 [24]。TD-Gammon 使用了类似于 Q-learning 的无模型强化学习算法，并使用具有一个隐藏层的多层感知机来逼近价值函数。

> 注：实际上 TD-Gammon 逼近的是状态价值函数 $V(s)$ 而非动作价值函数 $Q(s, a)$，且直接从自我对弈的在策略（On-Policy）游戏中学习。

然而，早期尝试沿用 TD-Gammon 方法的工作——包括将相同方法应用于国际象棋、围棋和跳棋——并不太成功。这导致了一种普遍的看法，即 TD-Gammon 方法是一个仅在西洋双陆棋中有效的特例，这可能是因为骰子的随机性有助于探索状态空间，也使得价值函数特别平滑 [19]。

此外，研究表明将无模型强化学习算法（如 Q-learning）与非线性函数逼近器结合 [25]，或者与离策略学习结合 [1]，可能导致 Q 网络发散。此后，强化学习领域的大部分工作集中于具有更好收敛保证的线性函数逼近器 [25]。

近年来，将深度学习与强化学习结合的兴趣重新复苏。深度神经网络已被用于估计环境 $\mathcal{E}$；受限玻尔兹曼机被用于估计价值函数 [21] 或策略 [9]。此外，Q-learning 的发散问题已被梯度时序差分（Gradient Temporal-Difference）方法部分解决。这些方法在使用非线性函数逼近器评估固定策略时已被证明收敛 [14]；或者在使用 Q-learning 的受限变体进行线性函数逼近的控制策略学习时收敛 [15]。然而，这些方法尚未扩展到非线性控制。

与我们的方法最相似的先前工作可能是神经拟合 Q 迭代（Neural Fitted Q-Learning, NFQ）[20]。NFQ 使用 RPROP 算法更新 Q 网络的参数来优化公式 2 中的损失函数序列。然而，它使用了每次迭代的计算成本与数据集大小成正比的批量更新，而我们考虑的是具有低常数每次迭代成本且可扩展到大数据集的随机梯度更新。NFQ 也曾通过先使用深度自编码器（Deep Autoencoder）学习任务的低维表示，然后将 NFQ 应用于此表示，成功应用于仅使用视觉输入的简单现实世界控制任务 [12]。相比之下，我们的方法端到端地应用强化学习，直接从视觉输入出发，因此可能学习到与区分动作价值直接相关的特征。Q-learning 此前也曾与经验回放和简单神经网络结合 [13]，但同样是从低维状态而非原始视觉输入开始。

Bellemare 等人 [3] 引入了将 Atari 2600 模拟器作为强化学习平台的方法，他们应用了标准强化学习算法，配合线性函数逼近和通用视觉特征。随后，通过使用更多特征并采用拔河哈希（Tug-of-War Hashing）将特征随机投影到更低维空间，结果得到了改进 [2]。HyperNEAT 进化架构 [8] 也被应用于 Atari 平台，用于为每个不同游戏分别进化代表该游戏策略的神经网络。当使用模拟器的重置功能针对确定性序列反复训练时，这些策略能够利用若干 Atari 游戏的设计缺陷。

---

## 4 深度强化学习

计算机视觉和语音识别的最新突破依赖于在超大训练集上高效训练深度神经网络。最成功的方法直接从原始输入进行训练，使用基于随机梯度下降的轻量级更新。通过向深度神经网络馈入足够的数据，通常可以学到比手工特征更好的表示 [11]。这些成功激励了我们的强化学习方法。我们的目标是将强化学习算法连接到一个直接处理 RGB 图像的深度神经网络，并通过随机梯度更新高效地处理训练数据。

Tesauro 的 TD-Gammon 架构为这一方法提供了起点。该架构根据从算法与环境交互中获得的在策略经验样本 $s_t, a_t, r_t, s_{t+1}, a_{t+1}$ 来更新估计价值函数的网络参数。由于该方法在 20 年前就能够击败最优秀的人类西洋双陆棋玩家，人们自然会好奇，二十年的硬件进步加上现代深度神经网络架构和可扩展的强化学习算法，是否能够产生显著的进展。

与 TD-Gammon 及类似的在线方法不同，我们利用了一种称为经验回放 [13] 的技术，将智能体在每个时间步的经验 $e_t = (s_t, a_t, r_t, s_{t+1})$ 存储在数据集 $\mathcal{D} = e_1, \ldots, e_N$ 中，该数据集汇集了许多回合（Episode）的数据形成回放记忆（Replay Memory）。在算法的内循环中，我们对从存储样本池中随机抽取的经验样本 $e \sim \mathcal{D}$ 应用 Q-learning 更新或小批量（Minibatch）更新。完成经验回放后，智能体根据 $\epsilon$-贪心策略选择并执行动作。由于将任意长度的历史作为神经网络输入可能存在困难，我们的 Q 函数使用由函数 $\phi$ 产生的固定长度历史表示。完整算法称为深度 Q-learning（Deep Q-Learning），见算法 1。

### 算法 1：带经验回放的深度 Q-learning

```
初始化回放记忆 D，容量为 N
初始化动作价值函数 Q，权重随机
for episode = 1, M do
    初始化序列 s_1 = {x_1}，预处理序列 φ_1 = φ(s_1)
    for t = 1, T do
        以概率 ε 选择随机动作 a_t
        否则选择 a_t = max_a Q*(φ(s_t), a; θ)
        在模拟器中执行动作 a_t，观测奖励 r_t 和图像 x_{t+1}
        设 s_{t+1} = s_t, a_t, x_{t+1}，预处理 φ_{t+1} = φ(s_{t+1})
        将转移 (φ_t, a_t, r_t, φ_{t+1}) 存入 D
        从 D 中随机采样小批量转移 (φ_j, a_j, r_j, φ_{j+1})
        设 y_j = r_j                            （若 φ_{j+1} 为终止状态）
              r_j + γ max_{a'} Q(φ_{j+1}, a'; θ) （若 φ_{j+1} 为非终止状态）
        对 (y_j - Q(φ_j, a_j; θ))^2 按公式 3 执行梯度下降步骤
    end for
end for
```

该方法相比标准在线 Q-learning [23] 具有若干优势。第一，每一步经验可能在多次权重更新中被使用，从而提高了数据效率。第二，直接从连续样本学习效率低下，因为样本之间存在强相关性；随机化样本打破了这些相关性，从而降低了更新的方差。第三，在策略学习时，当前参数决定了下一个用于训练的数据样本。例如，如果最大化动作是向左移动，则训练样本将被左侧的样本主导；如果最大化动作转为向右，则训练分布也会切换。容易看出不良的反馈循环可能产生，参数可能陷入较差的局部最小值甚至发散 [25]。通过使用经验回放，行为分布在许多先前状态上取平均，从而平滑学习过程，避免参数的振荡或发散。注意，使用经验回放学习时，必须采用离策略学习（因为当前参数与生成样本时使用的参数不同），这也正是选择 Q-learning 的动机。

在实践中，我们的算法仅在回放记忆中存储最近 $N$ 个经验元组，并在执行更新时从 $\mathcal{D}$ 中均匀随机采样。这种方法在某些方面存在局限性，因为记忆缓冲区不区分重要转移，且由于有限的记忆大小 $N$，总是用最近的转移覆盖旧数据。类似地，均匀采样赋予回放记忆中所有转移同等的重要性。更复杂的采样策略可以强调那些我们能从中学到最多的转移，类似于优先扫描（Prioritized Sweeping）[17]。

### 4.1 预处理与模型架构

直接处理原始 Atari 帧（210 $\times$ 160 像素，128 色调色板）在计算上要求较高，因此我们应用了基本的预处理步骤以降低输入维度。首先将原始帧从 RGB 表示转换为灰度图，然后下采样至 110 $\times$ 84 的图像。最终输入表示通过裁剪图像的 84 $\times$ 84 区域获得，该区域大致捕获了游戏区域。最终裁剪阶段仅因为我们使用了 [11] 中的 GPU 2D 卷积实现（要求方形输入）而需要。在本文实验中，算法 1 中的函数 $\phi$ 对历史的最后 4 帧应用此预处理，并将它们堆叠以产生 Q 函数的输入。

使用神经网络参数化 $Q$ 有几种可能的方式。由于 $Q$ 将历史-动作对映射到其 Q 值的标量估计，一些先前的方法将历史和动作都作为神经网络的输入 [20, 12]。这种架构的主要缺点是需要单独的前向传播来计算每个动作的 Q 值，导致成本随动作数量线性增长。我们使用的架构为每个可能的动作设置了单独的输出单元，仅将状态表示作为网络输入。输出对应于输入状态下各个动作的预测 Q 值。这种架构的主要优势在于只需网络的一次前向传播即可计算给定状态下所有可能动作的 Q 值。

以下是用于所有七个 Atari 游戏的具体架构。网络输入为 $\phi$ 产生的 84 $\times$ 84 $\times$ 4 图像。第一个隐藏层使用 16 个 8 $\times$ 8 的滤波器、步幅为 4 对输入图像进行卷积，并应用修正线性单元（Rectifier Nonlinearity）[10, 18]。第二个隐藏层使用 32 个 4 $\times$ 4 的滤波器、步幅为 2 进行卷积，同样后接修正线性单元。最后一个隐藏层是全连接层，由 256 个修正线性单元组成。输出层是全连接线性层，每个有效动作对应一个输出。我们考虑的游戏中有效动作数量在 4 到 18 之间。我们将使用此方法训练的卷积网络称为深度 Q 网络（Deep Q-Networks, DQN）。

---

## 5 实验

我们在七个流行的 ATARI 游戏上进行了实验——Beam Rider、Breakout、Enduro、Pong、Q\*bert、Seaquest、Space Invaders。所有七个游戏使用相同的网络架构、学习算法和超参数设置，表明我们的方法足够鲁棒，无需纳入游戏特定信息即可在多种游戏上工作。虽然我们在真实且未修改的游戏上评估智能体，但仅在训练期间对游戏的奖励结构做了一处修改：由于不同游戏的分数尺度差异很大，我们将所有正奖励固定为 1，所有负奖励固定为 $-1$，0 奖励保持不变。以这种方式裁剪奖励限制了误差导数的尺度，使得在多个游戏中使用相同的学习率成为可能。同时，这可能影响智能体的性能，因为它无法区分不同量级的奖励。

实验中，我们使用了 RMSProp 算法，小批量大小为 32。训练期间的行为策略为 $\epsilon$-贪心策略，$\epsilon$ 在前一百万帧内从 1 线性退火至 0.1，此后固定为 0.1。总共训练了一千万帧，使用了包含最近一百万帧的回放记忆。

沿用先前玩 Atari 游戏的方法，我们还使用了简单的跳帧技术 [3]。具体而言，智能体每 $k$ 帧而非每帧观测并选择动作，在跳过的帧上重复其上一个动作。由于模拟器前进一步所需的计算远少于智能体选择动作，这一技术使智能体能够在不显著增加运行时间的情况下玩大约 $k$ 倍多的游戏。所有游戏使用 $k = 4$，但 Space Invaders 除外——我们注意到使用 $k = 4$ 会因激光闪烁的周期而使激光不可见。我们使用 $k = 3$ 来使激光可见，这是所有游戏之间唯一的超参数差异。

### 5.1 训练与稳定性

在监督学习中，可以通过在训练集和验证集上评估模型来轻松跟踪训练期间的性能。然而在强化学习中，准确评估智能体在训练期间的进展可能具有挑战性。由于我们的评估指标（如 [3] 所建议的）是智能体在一个回合或游戏中收集的总奖励（在多个游戏上取平均），我们在训练期间定期计算该指标。平均总奖励指标往往非常嘈杂，因为策略权重的微小变化可能导致策略访问的状态分布发生巨大变化。

> **图 2：** 左侧两张图分别展示了 Breakout 和 Seaquest 在训练期间每回合的平均奖励。统计数据通过运行 $\epsilon = 0.05$ 的 $\epsilon$-贪心策略 10000 步来计算。右侧两张图分别展示了 Breakout 和 Seaquest 在一组保留状态上的平均最大预测动作价值。一个 epoch 对应 50000 次小批量权重更新，约 30 分钟的训练时间。

另一个更稳定的指标是策略估计的动作价值函数 $Q$，它提供了智能体从任何给定状态出发遵循其策略所能获得的折扣奖励估计。我们在训练开始前通过运行随机策略收集一组固定状态，并跟踪这些状态的最大预测 $Q$ 值的平均值。图 2 右侧两张图显示，平均预测 $Q$ 的增长比智能体获得的平均总奖励平滑得多，其他五个游戏的相同指标也产生了类似的平滑曲线。除了在训练期间观察到预测 $Q$ 的相对平滑改进之外，我们在所有实验中都未遇到发散问题。这表明，尽管缺乏理论收敛保证，我们的方法能够使用强化学习信号和随机梯度下降以稳定的方式训练大型神经网络。

### 5.2 价值函数可视化

> **图 3：** 最左侧的图展示了 Seaquest 游戏 30 帧片段的预测价值函数。三个截图分别对应标记为 A、B、C 的帧。

图 3 展示了在 Seaquest 游戏上学到的价值函数的可视化。该图显示，当敌人出现在屏幕左侧时，预测值跳升（A 点）。然后智能体向敌人发射鱼雷，当鱼雷即将击中敌人时预测值达到峰值（B 点）。最后，在敌人消失后，价值回落到大约其原始值（C 点）。图 3 证明我们的方法能够学习到价值函数如何随一个较为复杂的事件序列演变。

### 5.3 主要评估

我们将结果与强化学习文献中表现最好的方法进行了比较 [3, 4]。标记为 Sarsa 的方法使用 Sarsa 算法在为 Atari 任务手工设计的多个不同特征集上学习线性策略，我们报告了最佳特征集的分数 [3]。Contingency 使用了与 Sarsa 相同的基本方法，但用学到的屏幕中受智能体控制部分的表示来增强特征集 [4]。注意，这两种方法都通过使用背景减除和将 128 种颜色中的每一种作为单独通道来纳入了关于视觉问题的大量先验知识。由于许多 Atari 游戏为每种类型的对象使用一种不同的颜色，将每种颜色作为单独通道可以类似于为每种对象类型生成单独的二值图以编码其存在。相比之下，我们的智能体仅接收原始 RGB 截图作为输入，必须自行学习检测对象。

除了学习型智能体之外，我们还报告了人类专家游戏玩家和均匀随机选择动作的策略的分数。人类表现是在每个游戏约两小时后取得的中位奖励。注意我们报告的人类分数远高于 Bellemare 等人 [3] 中的分数。对于学习方法，我们遵循 Bellemare 等人 [3, 5] 使用的评估策略，报告以 $\epsilon = 0.05$ 运行 $\epsilon$-贪心策略固定步数所获得的平均分数。

**表 1：** 各种学习方法的比较结果

| 方法 | B. Rider | Breakout | Enduro | Pong | Q\*bert | Seaquest | S. Invaders |
|------|----------|----------|--------|------|---------|----------|-------------|
| Random | 354 | 1.2 | 0 | -20.4 | 157 | 110 | 179 |
| Sarsa [3] | 996 | 5.2 | 129 | -19 | 614 | 665 | 271 |
| Contingency [4] | 1743 | 6 | 159 | -17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | -3 | 18900 | 28010 | 3690 |
| HNeat Best [8] | 3616 | 52 | 106 | 19 | 1800 | 920 | 1720 |
| HNeat Pixel [8] | 1332 | 4 | 91 | -16 | 1325 | 800 | 1145 |
| DQN Best | 5184 | 225 | 661 | 21 | 4500 | 1740 | 1075 |

> 上半部分比较了以 $\epsilon = 0.05$ 运行 $\epsilon$-贪心策略固定步数的各种学习方法的平均总奖励。下半部分报告了 HNeat 和 DQN 单次最佳回合的结果。HNeat 产生确定性策略，总是获得相同分数，而 DQN 使用 $\epsilon = 0.05$ 的 $\epsilon$-贪心策略。

我们的方法（标记为 DQN）在所有七个游戏上以显著优势超越了其他学习方法，且几乎未纳入关于输入的先验知识。

我们还在表 1 的最后三行中包含了与 [8] 中进化策略搜索方法的比较。我们报告了该方法的两组结果。HNeat Best 分数反映了使用手工设计的对象检测器算法（输出 Atari 屏幕上对象的位置和类型）获得的结果。HNeat Pixel 分数通过使用 Atari 模拟器的特殊 8 色通道表示获得，该表示在每个通道表示一个对象标签图。这种方法严重依赖于找到代表成功利用的确定性状态序列。以这种方式学到的策略不太可能泛化到随机扰动；因此该算法仅在最高分的单次回合上进行评估。相比之下，我们的算法在 $\epsilon$-贪心控制序列上评估，因此必须在广泛的可能情况中泛化。尽管如此，我们表明在所有游戏上（除 Space Invaders 外），不仅我们的最大值评估结果（第 8 行），甚至我们的平均结果（第 4 行）也取得了更好的表现。

最后，我们表明我们的方法在 Breakout、Enduro 和 Pong 上超过了人类专家玩家的表现，并在 Beam Rider 上接近人类水平。在 Q\*bert、Seaquest 和 Space Invaders 上，我们距人类表现还有较大差距——这些游戏更具挑战性，因为它们要求网络找到跨越长时间尺度的策略。

---

## 6 结论

本文提出了一种新的用于强化学习的深度学习模型，并展示了其仅使用原始像素作为输入即可掌握 Atari 2600 电脑游戏的复杂控制策略的能力。我们还提出了在线 Q-learning 的一种变体，将随机小批量更新与经验回放记忆相结合，以简化深度网络在强化学习中的训练。我们的方法在测试的七个游戏中的六个上取得了当时最优的结果，且无需调整架构或超参数。

---

## 参考文献

[1] Leemon Baird. Residual algorithms: Reinforcement learning with function approximation. In Proceedings of the 12th International Conference on Machine Learning (ICML 1995), pages 30–37. Morgan Kaufmann, 1995.

[2] Marc Bellemare, Joel Veness, and Michael Bowling. Sketch-based linear value function approximation. In Advances in Neural Information Processing Systems 25, pages 2222–2230, 2012.

[3] Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253–279, 2013.

[4] Marc G Bellemare, Joel Veness, and Michael Bowling. Investigating contingency awareness using atari 2600 games. In AAAI, 2012.

[5] Marc G. Bellemare, Joel Veness, and Michael Bowling. Bayesian learning of recursively factored environments. In Proceedings of the Thirtieth International Conference on Machine Learning (ICML 2013), pages 1211–1219, 2013.

[6] George E. Dahl, Dong Yu, Li Deng, and Alex Acero. Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition. Audio, Speech, and Language Processing, IEEE Transactions on, 20(1):30–42, January 2012.

[7] Alex Graves, Abdel-rahman Mohamed, and Geoffrey E. Hinton. Speech recognition with deep recurrent neural networks. In Proc. ICASSP, 2013.

[8] Matthew Hausknecht, Risto Miikkulainen, and Peter Stone. A neuro-evolution approach to general atari game playing. 2013.

[9] Nicolas Heess, David Silver, and Yee Whye Teh. Actor-critic reinforcement learning with energy-based policies. In European Workshop on Reinforcement Learning, page 43, 2012.

[10] Kevin Jarrett, Koray Kavukcuoglu, MarcAurelio Ranzato, and Yann LeCun. What is the best multi-stage architecture for object recognition? In Proc. International Conference on Computer Vision and Pattern Recognition (CVPR 2009), pages 2146–2153. IEEE, 2009.

[11] Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1106–1114, 2012.

[12] Sascha Lange and Martin Riedmiller. Deep auto-encoder neural networks in reinforcement learning. In Neural Networks (IJCNN), The 2010 International Joint Conference on, pages 1–8. IEEE, 2010.

[13] Long-Ji Lin. Reinforcement learning for robots using neural networks. Technical report, DTIC Document, 1993.

[14] Hamid Maei, Csaba Szepesvari, Shalabh Bhatnagar, Doina Precup, David Silver, and Rich Sutton. Convergent Temporal-Difference Learning with Arbitrary Smooth Function Approximation. In Advances in Neural Information Processing Systems 22, pages 1204–1212, 2009.

[15] Hamid Maei, Csaba Szepesvári, Shalabh Bhatnagar, and Richard S. Sutton. Toward off-policy learning control with function approximation. In Proceedings of the 27th International Conference on Machine Learning (ICML 2010), pages 719–726, 2010.

[16] Volodymyr Mnih. Machine Learning for Aerial Image Labeling. PhD thesis, University of Toronto, 2013.

[17] Andrew Moore and Chris Atkeson. Prioritized sweeping: Reinforcement learning with less data and less real time. Machine Learning, 13:103–130, 1993.

[18] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML 2010), pages 807–814, 2010.

[19] Jordan B. Pollack and Alan D. Blair. Why did td-gammon work. In Advances in Neural Information Processing Systems 9, pages 10–16, 1996.

[20] Martin Riedmiller. Neural fitted q iteration–first experiences with a data efficient neural reinforcement learning method. In Machine Learning: ECML 2005, pages 317–328. Springer, 2005.

[21] Brian Sallans and Geoffrey E. Hinton. Reinforcement learning with factored states and actions. Journal of Machine Learning Research, 5:1063–1088, 2004.

[22] Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, and Yann LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In Proc. International Conference on Computer Vision and Pattern Recognition (CVPR 2013). IEEE, 2013.

[23] Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, 1998.

[24] Gerald Tesauro. Temporal difference learning and td-gammon. Communications of the ACM, 38(3):58–68, 1995.

[25] John N Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. Automatic Control, IEEE Transactions on, 42(5):674–690, 1997.

[26] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3-4):279–292, 1992.
