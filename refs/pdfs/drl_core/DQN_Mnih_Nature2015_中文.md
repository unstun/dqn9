# 通过深度强化学习实现人类水平的控制

**doi:10.1038/nature14236**

**作者：** Volodymyr Mnih\*, Koray Kavukcuoglu\*, David Silver\*, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis

*\* 这些作者对本工作贡献相同。*

Google DeepMind, 5 New Street Square, London EC4A 3TW, UK.

**发表于：** Nature, Vol 518, 2015年2月26日

---

## 摘要

强化学习（Reinforcement Learning, RL）理论提供了一种关于智能体如何优化其对环境控制的规范性描述[1]，该理论深深植根于心理学[2]和神经科学[3]对动物行为的研究视角。然而，要在接近现实世界复杂度的情境中成功使用强化学习，智能体面临着一项艰巨的任务：它们必须从高维感官输入中提取环境的高效表征，并利用这些表征将过去的经验泛化到新情境中。值得注意的是，人类和其他动物似乎通过强化学习与层次化感官处理系统[4,5]的和谐结合来解决这一问题，前者的证据来自大量神经数据，揭示了多巴胺能神经元发出的相位信号与时序差分强化学习算法（Temporal Difference Reinforcement Learning）之间的显著相似性[3]。虽然强化学习智能体已在多个领域取得了一些成功[6-8]，但其适用性此前仅限于可以手工设计有效特征的领域，或具有完全可观测、低维状态空间的领域。本文利用训练深度神经网络（Deep Neural Networks）的最新进展[9-11]，开发了一种新型人工智能体，称为深度Q网络（Deep Q-Network, DQN），它能够使用端到端强化学习（End-to-End Reinforcement Learning）直接从高维感官输入中学习成功的策略。我们在经典的Atari 2600游戏[12]这一具有挑战性的领域中测试了该智能体。我们证明，深度Q网络智能体仅接收像素和游戏分数作为输入，就能够在一组49个游戏中超越所有先前算法的性能，并达到与专业人类游戏测试员相当的水平，且在所有游戏中使用相同的算法、网络架构和超参数。这项工作弥合了高维感官输入与动作之间的鸿沟，产生了第一个能够学习在多种具有挑战性的任务中表现出色的人工智能体。

---

## 1 引言

我们着手创建一种能够在多种具有挑战性的任务上发展出广泛能力的单一算法——这是通用人工智能（General Artificial Intelligence）的核心目标[13]，此前的努力未能实现[8,14,15]。为此，我们开发了一种新型智能体——深度Q网络（DQN），它能够将强化学习与一类被称为深度神经网络的人工神经网络[16]相结合。值得注意的是，深度神经网络的最新进展[9-11]使用多层节点逐步构建数据越来越抽象的表征，使人工神经网络能够直接从原始感官数据中学习诸如物体类别等概念。我们使用了一种特别成功的架构——深度卷积网络（Deep Convolutional Network）[17]，它使用层次化的平铺卷积滤波器来模拟感受野（Receptive Fields）的效果——灵感来自Hubel和Wiesel关于早期视觉皮层前馈处理的开创性工作[18]——从而利用图像中存在的局部空间相关性，并内建对自然变换（如视角或尺度变化）的鲁棒性。

我们考虑的任务中，智能体通过一系列观测、动作和奖励与环境交互。智能体的目标是以最大化累积未来奖励的方式选择动作。更形式化地说，我们使用深度卷积神经网络来近似最优动作-价值函数（Optimal Action-Value Function）：

$$Q^*(s,a) = \max_{\pi} \mathbb{E}\left[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t = s, a_t = a, \pi\right]$$

即在做出观测 $s$ 并采取动作 $a$ 之后，通过行为策略 $\pi = P(a|s)$，以折扣因子 $\gamma$ 在每个时间步 $t$ 折扣奖励 $r_t$ 所能获得的最大奖励总和[19]。

已知当使用非线性函数逼近器（如神经网络）来表示动作-价值函数（也称为Q函数）时，强化学习是不稳定的甚至可能发散[20]。这种不稳定性有几个原因：观测序列中存在的相关性，Q的微小更新可能显著改变策略从而改变数据分布，以及动作价值Q与目标值 $r + \gamma \max_{a'} Q(s', a')$ 之间的相关性。

我们用Q学习（Q-Learning）的一种新变体来解决这些不稳定性，该变体使用了两个关键思想。第一，我们使用了一种受生物学启发的机制，称为经验回放（Experience Replay）[21-23]，它对数据进行随机化，从而消除观测序列中的相关性并平滑数据分布的变化。第二，我们使用了一种迭代更新方法，将动作价值Q朝着仅周期性更新的目标值调整，从而减少与目标的相关性。

虽然存在其他稳定的方法用于在强化学习环境中训练神经网络，如神经拟合Q迭代（Neural Fitted Q-Iteration）[24]，但这些方法需要在数百次迭代中从头重复训练网络。因此，这些方法不同于我们的算法，效率过低，无法成功用于大型神经网络。我们使用图1所示的深度卷积神经网络参数化近似价值函数 $Q(s,a;\theta_i)$，其中 $\theta_i$ 是Q网络在迭代 $i$ 时的参数（即权重）。为执行经验回放，我们在每个时间步 $t$ 将智能体的经验 $e_t = (s_t, a_t, r_t, s_{t+1})$ 存储在数据集 $D_t = \{e_1, \ldots, e_t\}$ 中。在学习过程中，我们对从存储样本池中均匀随机抽取的经验样本（或小批量）$(s, a, r, s') \sim U(D)$ 应用Q学习更新。迭代 $i$ 时的Q学习更新使用以下损失函数：

$$L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[\left(r + \gamma \max_{a'} Q(s', a'; \theta_i^{-}) - Q(s, a; \theta_i)\right)^2\right]$$

其中 $\gamma$ 是决定智能体视野的折扣因子，$\theta_i$ 是Q网络在迭代 $i$ 时的参数，$\theta_i^{-}$ 是用于在迭代 $i$ 时计算目标的网络参数。目标网络参数 $\theta_i^{-}$ 仅每 $C$ 步用Q网络参数 $\theta_i$ 更新一次，在各次更新之间保持固定。

---

## 2 实验评估

为评估我们的DQN智能体，我们利用了Atari 2600平台，该平台提供了多样化的任务（$n = 49$），旨在对人类玩家具有难度和吸引力。我们在所有游戏中使用了相同的网络架构、超参数值（见扩展数据表1）和学习过程——以高维数据（$210 \times 160$ 彩色视频，60 Hz）作为输入——以证明我们的方法仅基于感官输入，几乎不需要先验知识（即仅知道输入数据是视觉图像，以及每个游戏中可用的动作数量，但不知道动作的对应关系），就能稳健地在多种游戏中学习成功的策略。值得注意的是，我们的方法能够使用强化学习信号和随机梯度下降以稳定的方式训练大型神经网络——两个学习指标（智能体的平均每回合得分和平均预测Q值）的时间演变说明了这一点（见图2）。

我们将DQN与强化学习文献中表现最佳的方法在49个游戏上进行了比较[12,15]。除了学习型智能体外，我们还报告了在受控条件下进行游戏的专业人类游戏测试员的得分以及均匀随机选择动作的策略的得分（扩展数据表2和图3，y轴上分别标记为100%（人类）和0%（随机））。我们的DQN方法在43个游戏中优于现有最佳强化学习方法，且未纳入其他方法所使用的任何关于Atari 2600游戏的额外先验知识。此外，我们的DQN智能体在整组49个游戏中达到了与专业人类游戏测试员相当的水平，在超过一半的游戏（29个）中达到了人类得分的75%以上（见图3和扩展数据表2）。在额外的仿真中（见扩展数据表3和4），我们通过分别禁用DQN智能体的各个核心组件——回放记忆（Replay Memory）、独立的目标Q网络（Separate Target Q-Network）和深度卷积网络架构——并展示其对性能的不利影响，证明了这些组件的重要性。

---

## 3 表征分析

我们接下来在太空侵略者（Space Invaders）游戏的背景下，通过使用一种为高维数据可视化开发的技术——t-SNE[25]——来检验DQN所学习的、支撑智能体成功表现的表征（图4）。正如预期，t-SNE算法倾向于将感知上相似的状态的DQN表征映射到相邻的点。有趣的是，我们还发现t-SNE算法为在期望奖励上接近但在感知上不相似的状态生成了相似的嵌入（图4，右下、左上和中间），这与网络能够从高维感官输入中学习支持适应性行为的表征这一概念一致。此外，我们还表明DQN学习的表征能够泛化到由其自身策略之外的策略生成的数据——在仿真中，我们将人类和智能体游戏过程中经历的游戏状态作为输入呈现给网络，记录最后一个隐藏层的表征，并可视化t-SNE算法生成的嵌入（扩展数据图1）。扩展数据图2进一步说明了DQN学习的表征如何使其能够准确预测状态和动作价值。

值得注意的是，DQN擅长的游戏在性质上极为多样，从横向卷轴射击游戏（River Raid）到拳击游戏（Boxing）和三维赛车游戏（Enduro）。确实，在某些游戏中，DQN能够发现相对长期的策略，例如在Breakout中：智能体学习了最优策略，即先在墙壁侧面挖一条隧道，让球绕到墙壁后面以摧毁大量砖块。然而，需要更长时间规划策略的游戏对于包括DQN在内的所有现有智能体仍然构成重大挑战，例如Montezuma's Revenge。

---

## 4 讨论

在这项工作中，我们证明了单一架构能够在一系列不同环境中成功学习控制策略，仅需极少的先验知识，仅接收像素和游戏分数作为输入，并在每个游戏中使用相同的算法、网络架构和超参数，仅获得人类玩家所能获得的输入信息。与先前工作[24,26]不同，我们的方法采用"端到端"强化学习，利用奖励持续塑造卷积网络中的表征，使其朝着有利于价值估计的环境显著特征发展。这一原理借鉴了神经生物学证据，表明感知学习过程中的奖励信号可能影响灵长类视觉皮层中表征的特性[27,28]。值得注意的是，强化学习与深度网络架构的成功整合关键依赖于我们引入的回放算法[21-23]，该算法涉及近期经历的转换的存储和表征。汇聚证据表明，海马体（Hippocampus）可能支持此类过程在哺乳动物大脑中的物理实现，在离线期间（例如清醒休息时）对近期经历轨迹的时间压缩重激活[21,22]提供了一种假设性机制，通过与基底神经节（Basal Ganglia）的交互来高效更新价值函数[22]。未来，探索将经验回放内容偏向于显著事件的潜在用途将很重要，这一现象是经验观察到的海马回放的特征[29]，并与强化学习中"优先扫描"（Prioritized Sweeping）的概念相关[30]。总而言之，我们的工作展示了利用最先进的机器学习技术与受生物学启发的机制相结合来创建能够学习掌握多种具有挑战性任务的智能体的强大力量。

---

## 方法

### 预处理

直接处理原始Atari 2600帧——$210 \times 160$ 像素的图像，具有128色调色板——在计算和内存需求方面可能非常大。我们应用了一个基本预处理步骤，旨在降低输入维度并处理Atari 2600模拟器的一些伪影。首先，为编码单帧，我们取当前编码帧和前一帧中每个像素颜色值的最大值。这对于消除游戏中的闪烁是必要的，因为某些物体仅出现在偶数帧中而其他物体仅出现在奇数帧中，这是由Atari 2600一次只能显示有限数量精灵造成的伪影。其次，我们从RGB帧中提取Y通道（即亮度通道），并将其缩放到 $84 \times 84$。下面算法1中描述的函数 $\phi$ 将此预处理应用于最近的 $m$ 帧并将它们堆叠以生成Q函数的输入，其中 $m = 4$，但该算法对 $m$ 的不同值（例如3或5）具有鲁棒性。

**代码可用性：** 源代码可在 https://sites.google.com/a/deepmind.com/dqn 访问，仅限非商业用途。

### 模型架构

使用神经网络参数化Q有几种可能的方式。因为Q将历史-动作对映射到其Q值的标量估计，一些先前的方法[24,26]将历史和动作作为神经网络的输入。这种架构的主要缺点是需要单独的前向传播来计算每个动作的Q值，导致成本与动作数量线性增长。我们改用一种架构，其中每个可能的动作都有一个单独的输出单元，仅将状态表征作为神经网络的输入。输出对应于输入状态下各个动作的预测Q值。这种架构的主要优势是能够仅通过网络的一次前向传播就计算给定状态下所有可能动作的Q值。

如图1所示，确切的架构如下。神经网络的输入是由预处理映射 $\phi$ 生成的 $84 \times 84 \times 4$ 图像。第一个隐藏层将32个 $8 \times 8$ 步长为4的滤波器与输入图像卷积，并应用修正线性非线性激活函数（Rectifier Nonlinearity）[31,32]。第二个隐藏层将64个 $4 \times 4$ 步长为2的滤波器卷积，同样后接修正线性激活。第三个卷积层将64个 $3 \times 3$ 步长为1的滤波器卷积，后接修正线性激活。最后的隐藏层是全连接层，包含512个修正线性单元。输出层是全连接线性层，每个有效动作有一个输出。在我们考虑的游戏中，有效动作数量在4到18之间。

### 训练细节

我们在49个Atari 2600游戏上进行了实验。为每个游戏训练一个不同的网络：在所有游戏中使用相同的网络架构、学习算法和超参数设置（见扩展数据表1），表明我们的方法足够鲁棒，能够在多种游戏上工作，同时仅纳入极少的先验知识。虽然我们在未修改的游戏上评估智能体，但在训练期间我们对游戏的奖励结构做了一项更改。由于不同游戏的分数尺度差异很大，我们将所有正奖励裁剪为1，所有负奖励裁剪为-1，零奖励不变。这种奖励裁剪方式限制了误差导数的尺度，便于在多个游戏中使用相同的学习率。同时，这可能影响智能体的性能，因为它无法区分不同大小的奖励。对于有生命计数的游戏，Atari 2600模拟器还会发送游戏中剩余的生命数，用于在训练期间标记回合的结束。

在这些实验中，我们使用RMSProp算法，小批量大小为32。训练期间的行为策略为 $\epsilon$-贪心（$\epsilon$-Greedy），$\epsilon$ 在前一百万帧内从1.0线性退火到0.1，之后固定为0.1。我们总共训练了5000万帧（即约38天的游戏体验），使用了包含最近100万帧的回放记忆。

遵循先前在Atari 2600游戏上的方法，我们还使用了简单的跳帧技术（Frame-Skipping）[15]。具体而言，智能体每隔 $k$ 帧才观察和选择动作，在跳过的帧上重复其上一个动作。因为运行模拟器前进一步所需的计算远少于让智能体选择动作，所以这种技术使智能体能够在不显著增加运行时间的情况下大约多玩 $k$ 倍的游戏。我们在所有游戏中使用 $k = 4$。

所有超参数和优化参数的值是通过在Pong、Breakout、Seaquest、Space Invaders和Beam Rider游戏上进行非正式搜索选定的。由于高计算成本，我们没有进行系统的网格搜索。这些参数随后在所有其他游戏中保持固定。

我们的实验设置使用了以下极少的先验知识：输入数据由视觉图像组成（促使我们使用卷积深度网络）、游戏特定的分数（未做修改）、动作数量（但不包括其对应关系，例如"上"按钮的指定）以及生命计数。

### 评估过程

训练好的智能体通过以下方式评估：每个游戏玩30次，每次最多5分钟，使用不同的初始随机条件（"空操作"；见扩展数据表1）和 $\epsilon = 0.05$ 的 $\epsilon$-贪心策略。采用此过程以最大程度减少评估期间的过拟合。随机智能体作为基线比较，以10 Hz（即每6帧）随机选择动作，在中间帧上重复上一个动作。10 Hz大约是人类玩家选择"发射"按钮的最快频率，将随机智能体设为此频率可避免少数游戏中的虚假基线得分。我们也评估了以60 Hz（即每帧）选择动作的随机智能体的性能。这对结果影响极小：仅在6个游戏（Boxing、Breakout、Crazy Climber、Demon Attack、Krull和Robotank）中使归一化DQN性能变化超过5%，且在所有这些游戏中DQN都以相当大的优势超越了专家级人类。

专业人类测试员使用与智能体相同的模拟器引擎，并在受控条件下进行游戏。人类测试员不允许暂停、保存或重新加载游戏。与原始Atari 2600环境一样，模拟器以60 Hz运行且禁用了音频输出：因此人类玩家和智能体之间的感官输入是相同的。人类性能是每个游戏约20个回合的平均奖励，每个回合最长5分钟，在约2小时的练习之后进行。

### 算法

我们考虑智能体与环境（此处为Atari模拟器）通过一系列动作、观测和奖励进行交互的任务。在每个时间步，智能体从合法游戏动作集合 $\mathcal{A} = \{1, \ldots, K\}$ 中选择一个动作 $a_t$。该动作传递给模拟器并修改其内部状态和游戏分数。一般来说，环境可能是随机的。智能体不能观察模拟器的内部状态；相反，智能体观察来自模拟器的图像 $x_t \in \mathbb{R}^d$，这是代表当前屏幕的像素值向量。此外，它接收代表游戏分数变化的奖励 $r_t$。请注意，一般来说游戏分数可能依赖于整个先前的动作和观测序列；关于某个动作的反馈可能在经过数千个时间步后才被接收到。

由于智能体只能观察当前屏幕，该任务是部分可观测的（Partially Observed）[33]，许多模拟器状态在感知上是混叠的（即仅从当前屏幕 $x_t$ 不可能完全理解当前情况）。因此，动作和观测的序列 $s_t = x_1, a_1, x_2, \ldots, a_{t-1}, x_t$ 被输入算法，然后算法根据这些序列学习游戏策略。模拟器中的所有序列假设在有限数量的时间步内终止。这种形式化产生了一个大但有限的马尔可夫决策过程（Markov Decision Process, MDP），其中每个序列是一个不同的状态。因此，我们可以应用标准的MDP强化学习方法，简单地使用完整序列 $s_t$ 作为时间 $t$ 的状态表征。

智能体的目标是通过以最大化未来奖励的方式选择动作来与模拟器交互。我们采用标准假设，即未来奖励以每时间步 $\gamma$ 的因子折扣（$\gamma$ 始终设为0.99），并定义时间 $t$ 的未来折扣回报为 $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$，其中 $T$ 是游戏终止的时间步。我们定义最优动作-价值函数 $Q^*(s,a)$ 为在看到某个序列 $s$ 然后采取某个动作 $a$ 后，遵循任意策略所能获得的最大期望回报：

$$Q^*(s,a) = \max_{\pi} \mathbb{E}[R_t | s_t = s, a_t = a, \pi]$$

其中 $\pi$ 是将序列映射到动作（或动作分布）的策略。

最优动作-价值函数遵循一个重要的恒等式，称为贝尔曼方程（Bellman Equation）。其基于以下直觉：如果下一个时间步序列 $s'$ 对所有可能动作 $a'$ 的最优值 $Q^*(s',a')$ 已知，则最优策略是选择使 $r + \gamma Q^*(s',a')$ 的期望值最大化的动作 $a'$：

$$Q^*(s,a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

许多强化学习算法背后的基本思想是使用贝尔曼方程作为迭代更新来估计动作-价值函数：$Q_{i+1}(s,a) = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q_i(s',a') | s,a]$。这种值迭代算法当 $i \to \infty$ 时收敛到最优动作-价值函数 $Q_i \to Q^*$。在实践中，这种基本方法不可行，因为动作-价值函数是对每个序列单独估计的，没有任何泛化。相反，通常使用函数逼近器来估计动作-价值函数 $Q(s,a;\theta) \approx Q^*(s,a)$。在强化学习社区中，这通常是线性函数逼近器，但有时也会使用非线性函数逼近器，例如神经网络。我们将具有权重 $\theta$ 的神经网络函数逼近器称为Q网络（Q-Network）。Q网络可以通过在迭代 $i$ 时调整参数 $\theta_i$ 来减少贝尔曼方程中的均方误差进行训练，其中最优目标值 $r + \gamma \max_{a'} Q^*(s',a')$ 用近似目标值 $y = r + \gamma \max_{a'} Q(s',a';\theta_i^{-})$ 替代，使用来自先前某次迭代的参数 $\theta_i^{-}$。

这导致了在每次迭代 $i$ 时变化的损失函数序列 $L_i(\theta_i)$：

$$L_i(\theta_i) = \mathbb{E}_{s,a,r}\left[\left(\mathbb{E}_{s'}[y | s, a] - Q(s, a; \theta_i)\right)^2\right] = \mathbb{E}_{s,a,r,s'}\left[(y - Q(s,a;\theta_i))^2\right] + \mathbb{E}_{s,a,r}[\text{Var}_{s'}[y]]$$

请注意，目标依赖于网络权重；这与监督学习中在学习开始前就固定的目标不同。在优化的每个阶段，我们在优化第 $i$ 个损失函数 $L_i(\theta_i)$ 时将前一次迭代的参数 $\theta_i^{-}$ 保持固定，从而产生一系列定义良好的优化问题。最后一项是目标的方差，不依赖于我们当前优化的参数 $\theta_i$，因此可以忽略。对损失函数关于权重求微分，我们得到以下梯度：

$$\nabla_{\theta_i} L_i(\theta_i) = \mathbb{E}_{s,a,r,s'}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta_i^{-}) - Q(s, a; \theta_i)\right) \nabla_{\theta_i} Q(s, a; \theta_i)\right]$$

相比计算上述梯度中的完整期望，通过随机梯度下降优化损失函数在计算上通常更为便捷。熟悉的Q学习算法[19]可以在此框架中恢复，即在每个时间步后更新权重，用单个样本替代期望，并设 $\theta_i^{-} = \theta_{i-1}$。

请注意，该算法是无模型的（Model-Free）：它直接使用模拟器的样本解决强化学习任务，而不显式估计奖励和转移动力学 $P(r, s' | s, a)$。它也是离策略的（Off-Policy）：它学习贪心策略 $a = \arg\max_{a'} Q(s, a'; \theta)$，同时遵循确保充分探索状态空间的行为分布。在实践中，行为分布通常由 $\epsilon$-贪心策略选择，以概率 $1 - \epsilon$ 遵循贪心策略，以概率 $\epsilon$ 选择随机动作。

### 深度Q网络训练算法

训练深度Q网络的完整算法在算法1中给出。智能体根据基于Q的 $\epsilon$-贪心策略选择和执行动作。由于使用任意长度的历史作为神经网络的输入可能困难，我们的Q函数改为处理由上述函数 $\phi$ 生成的固定长度历史表征。该算法以两种方式修改标准在线Q学习，使其适合训练大型神经网络而不发散。

**第一，** 我们使用称为经验回放[23]的技术，在每个时间步将智能体的经验 $e_t = (s_t, a_t, r_t, s_{t+1})$ 存储在数据集 $D_t = \{e_1, \ldots, e_t\}$ 中，跨多个回合汇集到回放记忆中。在算法的内循环中，我们对从存储样本池中随机抽取的经验样本 $(s, a, r, s') \sim U(D)$ 应用Q学习更新或小批量更新。这种方法相比标准在线Q学习有几个优势。首先，经验的每一步可能用于多次权重更新，从而实现更高的数据效率。其次，直接从连续样本学习效率低下，因为样本之间存在强相关性；随机化样本打破了这些相关性，从而减少了更新的方差。第三，在线策略学习时，当前参数决定了参数接下来训练的下一个数据样本。例如，如果最大化动作是向左移动，则训练样本将被左侧的样本主导；如果最大化动作随后切换到向右，则训练分布也会切换。容易看出不需要的反馈回路可能会出现，参数可能陷入较差的局部最小值，甚至灾难性地发散[20]。通过使用经验回放，行为分布在其许多先前状态上取平均，平滑了学习过程并避免了参数的振荡或发散。请注意，通过经验回放学习时，必须进行离策略学习（因为我们的当前参数与生成样本时使用的参数不同），这促使我们选择Q学习。

在实践中，我们的算法仅在回放记忆中存储最近的 $N$ 个经验元组，在执行更新时从 $D$ 中均匀随机采样。这种方法在某些方面有局限性，因为记忆缓冲区不区分重要的转换，且由于有限的记忆大小 $N$ 总是用最近的转换覆盖。同样，均匀采样对回放记忆中的所有转换赋予相同的重要性。更复杂的采样策略可能强调我们能从中学到最多的转换，类似于优先扫描[30]。

**第二，** 对在线Q学习的第二项修改旨在进一步提高我们方法在神经网络上的稳定性，即使用一个独立的网络来生成Q学习更新中的目标 $y_j$。更确切地说，每 $C$ 次更新我们克隆网络Q得到目标网络 $\hat{Q}$，并使用 $\hat{Q}$ 在接下来的 $C$ 次更新中生成Q学习目标 $y_j$。这种修改使算法比标准在线Q学习更稳定，因为在标准方法中，增加 $Q(s_t, a_t)$ 的更新通常也会增加所有动作 $a$ 的 $Q(s_{t+1}, a)$，从而也增加目标 $y_j$，可能导致策略的振荡或发散。使用一组较旧的参数生成目标在Q的更新时刻和更新影响目标的时刻之间增加了延迟，使发散或振荡变得更加不可能。

我们还发现将更新中的误差项 $r + \gamma \max_{a'} Q(s',a';\theta_i^{-}) - Q(s,a;\theta_i)$ 裁剪到 $[-1, 1]$ 之间是有帮助的。因为绝对值损失函数 $|x|$ 对所有负值 $x$ 的导数为 $-1$，对所有正值 $x$ 的导数为 $1$，将平方误差裁剪到 $[-1, 1]$ 之间对应于对 $(-1, 1)$ 区间外的误差使用绝对值损失函数。这种形式的误差裁剪进一步提高了算法的稳定性。

### 算法1：带经验回放的深度Q学习

> 初始化回放记忆 $D$，容量为 $N$
>
> 用随机权重 $\theta$ 初始化动作-价值函数 $Q$
>
> 用权重 $\theta^{-} = \theta$ 初始化目标动作-价值函数 $\hat{Q}$
>
> **For** episode $= 1, M$ **do**
>
> &emsp; 初始化序列 $s_1 = \{x_1\}$，预处理序列 $\phi_1 = \phi(s_1)$
>
> &emsp; **For** $t = 1, T$ **do**
>
> &emsp;&emsp; 以概率 $\epsilon$ 选择随机动作 $a_t$
>
> &emsp;&emsp; 否则选择 $a_t = \arg\max_a Q(\phi(s_t), a; \theta)$
>
> &emsp;&emsp; 在模拟器中执行动作 $a_t$，观察奖励 $r_t$ 和图像 $x_{t+1}$
>
> &emsp;&emsp; 设 $s_{t+1} = s_t, a_t, x_{t+1}$，预处理 $\phi_{t+1} = \phi(s_{t+1})$
>
> &emsp;&emsp; 将转换 $(\phi_t, a_t, r_t, \phi_{t+1})$ 存入 $D$
>
> &emsp;&emsp; 从 $D$ 中随机采样一个小批量的转换 $(\phi_j, a_j, r_j, \phi_{j+1})$
>
> &emsp;&emsp; 设 $y_j = \begin{cases} r_j & \text{若回合在步骤 } j+1 \text{ 终止} \\ r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^{-}) & \text{否则} \end{cases}$
>
> &emsp;&emsp; 对 $(y_j - Q(\phi_j, a_j; \theta))^2$ 关于网络参数 $\theta$ 执行梯度下降步
>
> &emsp;&emsp; 每 $C$ 步重置 $\hat{Q} = Q$
>
> &emsp; **End For**
>
> **End For**

---

## 图表说明

### 图1：卷积神经网络的示意图

该图展示了DQN的网络架构。神经网络的输入是由预处理映射 $\phi$ 生成的 $84 \times 84 \times 4$ 图像，依次经过三个卷积层（蓝色蛇形线表示每个滤波器在输入图像上的滑动）和两个全连接层，每个有效动作对应一个输出。每个隐藏层后接修正线性非线性激活函数，即 $\max(0, x)$。

### 图2：训练曲线

追踪智能体的平均得分和平均预测动作价值的训练曲线。(a) 每个点是在Space Invaders上以 $\epsilon$-贪心策略（$\epsilon = 0.05$）运行520k帧后的每回合平均得分。(b) Seaquest的每回合平均得分。(c) Space Invaders上保留状态集的平均预测动作价值。曲线上的每个点是在保留状态集上计算的动作价值Q的平均值（注意由于奖励裁剪，Q值经过了缩放）。(d) Seaquest上的平均预测动作价值。

### 图3：DQN智能体与文献中最佳强化学习方法的比较

DQN的性能相对于专业人类游戏测试员（即100%水平）和随机游戏（即0%水平）进行了归一化。归一化性能以百分比表示，计算公式为：$100 \times (\text{DQN得分} - \text{随机得分}) / (\text{人类得分} - \text{随机得分})$。可以看出DQN在几乎所有游戏中优于竞争方法，并且在大多数游戏中达到了与专业人类游戏测试员大致相当或更优的水平（即75%或以上）。人类玩家和智能体均禁用了音频输出。误差条表示30次评估回合的标准差，每次使用不同的初始条件。

### 图4：t-SNE嵌入可视化

DQN在Space Invaders游戏过程中对游戏状态的最后隐藏层表征的二维t-SNE嵌入。该图通过让DQN智能体玩2小时的真实游戏时间，然后对DQN为每个经历的游戏状态分配的最后隐藏层表征运行t-SNE算法[25]生成。点根据DQN对相应游戏状态预测的状态值V（最大期望奖励）着色（从深红色（最高V）到深蓝色（最低V））。DQN智能体对满屏和接近满屏的状态预测较高的状态值，因为它已学会清除一屏会导致出现充满敌舰的新屏幕。部分清除的屏幕被赋予较低的状态值，因为可用的即时奖励较少。

### 扩展数据图1：人类与智能体游戏状态的t-SNE嵌入

DQN对Space Invaders中人类和智能体游戏经历的游戏状态的最后隐藏层表征的二维t-SNE嵌入。该图通过对人类（30分钟）和智能体（2小时）游戏经历的游戏状态的DQN最后隐藏层表征运行t-SNE算法生成。人类游戏（橙色点）和DQN游戏（蓝色点）的二维嵌入中存在相似结构，表明DQN学习的表征确实能够泛化到由其自身策略之外的策略生成的数据。

### 扩展数据图2：学习价值函数的可视化

在Breakout和Pong两个游戏上学习的价值函数的可视化。(a) Breakout游戏上学习的价值函数可视化：在时间点1和2，状态值预测约为17，智能体正在清除最底层的砖块。价值函数曲线的每个峰值对应于清除一块砖获得的奖励。在时间点3，智能体即将突破到顶层砖块，价值增加到约21。在时间点4，价值超过23，智能体已经突破。(b) Pong游戏上学习的动作-价值函数可视化：展示了智能体如何根据球的位置和运动方向准确预测不同动作的价值。

### 扩展数据表1：超参数列表及其值

所有超参数的值通过在Pong、Breakout、Seaquest、Space Invaders和Beam Rider游戏上进行非正式搜索选定。由于高计算成本，未进行系统的网格搜索，但可以预期系统地调整超参数值可能获得更好的结果。

### 扩展数据表2：DQN智能体与文献方法及专业人类游戏测试员的游戏得分比较

最佳线性学习器（Best Linear Learner）是线性函数逼近器在不同类型手工设计特征上获得的最佳结果[12]。Contingency（SARSA）智能体的数据来自文献[15]。最后一列的数字表示DQN相对于人类游戏测试员的性能百分比，即 $100 \times (\text{DQN得分} - \text{随机得分}) / (\text{人类得分} - \text{随机得分})$。

### 扩展数据表3：回放和独立目标Q网络的效果

DQN智能体使用标准超参数训练1000万帧，测试了开启/关闭回放、使用/不使用独立目标Q网络以及三种不同学习率的所有可能组合。每个智能体每250,000训练帧评估一次，进行135,000验证帧，报告最高平均回合得分。

### 扩展数据表4：DQN性能与线性函数逼近器的比较

将DQN智能体的性能与线性函数逼近器（即使用单个线性层替代卷积网络，结合回放和独立目标网络）在5个验证游戏上的性能进行比较。

---

## 参考文献

1. Sutton, R. & Barto, A. Reinforcement Learning: An Introduction (MIT Press, 1998).
2. Thorndike, E. L. Animal Intelligence: Experimental studies (Macmillan, 1911).
3. Schultz, W., Dayan, P. & Montague, P. R. A neural substrate of prediction and reward. Science 275, 1593–1599 (1997).
4. Serre, T., Wolf, L. & Poggio, T. Object recognition with features inspired by visual cortex. Proc. IEEE. Comput. Soc. Conf. Comput. Vis. Pattern. Recognit. 994–1000 (2005).
5. Fukushima, K. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biol. Cybern. 36, 193–202 (1980).
6. Tesauro, G. Temporal difference learning and TD-Gammon. Commun. ACM 38, 58–68 (1995).
7. Riedmiller, M., Gabel, T., Hafner, R. & Lange, S. Reinforcement learning for robot soccer. Auton. Robots 27, 55–73 (2009).
8. Diuk, C., Cohen, A. & Littman, M. L. An object-oriented representation for efficient reinforcement learning. Proc. Int. Conf. Mach. Learn. 240–247 (2008).
9. Bengio, Y. Learning deep architectures for AI. Foundations and Trends in Machine Learning 2, 1–127 (2009).
10. Krizhevsky, A., Sutskever, I. & Hinton, G. ImageNet classification with deep convolutional neural networks. Adv. Neural Inf. Process. Syst. 25, 1106–1114 (2012).
11. Hinton, G. E. & Salakhutdinov, R. R. Reducing the dimensionality of data with neural networks. Science 313, 504–507 (2006).
12. Bellemare, M. G., Naddaf, Y., Veness, J. & Bowling, M. The arcade learning environment: An evaluation platform for general agents. J. Artif. Intell. Res. 47, 253–279 (2013).
13. Legg, S. & Hutter, M. Universal Intelligence: a definition of machine intelligence. Minds Mach. 17, 391–444 (2007).
14. Genesereth, M., Love, N. & Pell, B. General game playing: overview of the AAAI competition. AI Mag. 26, 62–72 (2005).
15. Bellemare, M. G., Veness, J. & Bowling, M. Investigating contingency awareness using Atari 2600 games. Proc. Conf. AAAI. Artif. Intell. 864–871 (2012).
16. McClelland, J. L., Rumelhart, D. E. & Group, T. P. R. Parallel Distributed Processing: Explorations in the Microstructure of Cognition (MIT Press, 1986).
17. LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. Gradient-based learning applied to document recognition. Proc. IEEE 86, 2278–2324 (1998).
18. Hubel, D. H. & Wiesel, T. N. Shape and arrangement of columns in cat's striate cortex. J. Physiol. 165, 559–568 (1963).
19. Watkins, C. J. & Dayan, P. Q-learning. Mach. Learn. 8, 279–292 (1992).
20. Tsitsiklis, J. & Roy, B. V. An analysis of temporal-difference learning with function approximation. IEEE Trans. Automat. Contr. 42, 674–690 (1997).
21. McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychol. Rev. 102, 419–457 (1995).
22. O'Neill, J., Pleydell-Bouverie, B., Dupret, D. & Csicsvari, J. Play it again: reactivation of waking experience and memory. Trends Neurosci. 33, 220–229 (2010).
23. Lin, L.-J. Reinforcement learning for robots using neural networks. Technical Report, DTIC Document (1993).
24. Riedmiller, M. Neural fitted Q iteration - first experiences with a data efficient neural reinforcement learning method. Mach. Learn.: ECML, 3720, 317–328 (Springer, 2005).
25. Van der Maaten, L. J. P. & Hinton, G. E. Visualizing high-dimensional data using t-SNE. J. Mach. Learn. Res. 9, 2579–2605 (2008).
26. Lange, S. & Riedmiller, M. Deep auto-encoder neural networks in reinforcement learning. Proc. Int. Jt. Conf. Neural. Netw. 1–8 (2010).
27. Law, C.-T. & Gold, J. I. Reinforcement learning can account for associative and perceptual learning on a visual decision task. Nature Neurosci. 12, 655 (2009).
28. Sigala, N. & Logothetis, N. K. Visual categorization shapes feature selectivity in the primate temporal cortex. Nature 415, 318–320 (2002).
29. Bendor, D. & Wilson, M. A. Biasing the content of hippocampal replay during sleep. Nature Neurosci. 15, 1439–1444 (2012).
30. Moore, A. & Atkeson, C. Prioritized sweeping: reinforcement learning with less data and less real time. Mach. Learn. 13, 103–130 (1993).
31. Jarrett, K., Kavukcuoglu, K., Ranzato, M. A. & LeCun, Y. What is the best multi-stage architecture for object recognition? Proc. IEEE. Int. Conf. Comput. Vis. 2146–2153 (2009).
32. Nair, V. & Hinton, G. E. Rectified linear units improve restricted Boltzmann machines. Proc. Int. Conf. Mach. Learn. 807–814 (2010).
33. Kaelbling, L. P., Littman, M. L. & Cassandra, A. R. Planning and acting in partially observable stochastic domains. Artificial Intelligence 101, 99–134 (1994).
