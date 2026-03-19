# 基于深度强化学习的连续控制

**发表于 ICLR 2016 会议论文**

**作者：** Timothy P. Lillicrap\*, Jonathan J. Hunt\*, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra

**单位：** Google Deepmind, London, UK

---

## 摘要

我们将深度 Q 学习（Deep Q-Learning）成功背后的核心思想推广到连续动作域。我们提出了一种基于确定性策略梯度（deterministic policy gradient）的演员-评论家（actor-critic）无模型算法，能够在连续动作空间（continuous action spaces）上运行。使用相同的学习算法、网络架构和超参数，我们的算法能够稳健地解决超过 20 个模拟物理任务，包括倒立摆摆起（cartpole swing-up）、灵巧操控（dexterous manipulation）、腿式运动（legged locomotion）和汽车驾驶等经典问题。我们的算法能够找到性能与规划算法相当的策略，而后者拥有对环境动力学及其导数的完全访问权限。此外，我们还展示了在许多任务中，该算法能够"端到端"（end-to-end）学习策略，即直接从原始像素输入进行学习。

---

## 1 引言

人工智能领域的主要目标之一是从未经处理的高维感知输入中解决复杂任务。近年来，通过将深度学习在感知处理方面的进展 (Krizhevsky et al., 2012) 与强化学习相结合，取得了显著进展，其成果即"深度 Q 网络"（Deep Q Network, DQN）算法 (Mnih et al., 2015)，该算法能够使用未经处理的像素作为输入，在许多 Atari 视频游戏上达到人类水平的表现。为此，深度神经网络函数逼近器（deep neural network function approximators）被用于估计动作值函数（action-value function）。

然而，尽管 DQN 能够解决具有高维观测空间的问题，它只能处理离散且低维的动作空间。许多感兴趣的任务，尤其是物理控制任务，具有连续的（实值的）和高维的动作空间。DQN 无法直接应用于连续域，因为它依赖于找到使动作值函数最大化的动作，而在连续值情况下，这需要在每一步进行迭代优化。

将 DQN 等深度强化学习方法适应到连续域的一个直观方法是简单地离散化动作空间。然而这有许多局限性，最显著的是维度灾难（curse of dimensionality）：动作数量随自由度数量呈指数增长。例如，一个 7 自由度系统（如人类手臂），每个关节使用最粗糙的离散化 $a_i \in \{-k, 0, k\}$，将导致动作空间维度为 $3^7 = 2187$。对于需要精细动作控制的任务，情况更加糟糕，因为需要相应更细的离散化粒度，导致离散动作数量的爆炸式增长。如此大的动作空间难以高效探索，因此在这种情况下成功训练类 DQN 网络可能是不可行的。此外，简单的动作空间离散化会不必要地丢弃动作域结构的信息，而这些信息可能对解决许多问题至关重要。

在本工作中，我们提出了一种使用深度函数逼近器的无模型、离策略（off-policy）演员-评论家算法，能够在高维连续动作空间中学习策略。我们的工作基于确定性策略梯度（DPG）算法 (Silver et al., 2014)（其本身类似于 NFQCA (Hafner & Riedmiller, 2011)，类似的思想也可见于 (Prokhorov et al., 1997)）。然而如下所示，将该演员-评论家方法与神经网络函数逼近器的简单结合，对于具有挑战性的问题是不稳定的。

在此我们将演员-评论家方法与深度 Q 网络（DQN）近期成功的经验相结合 (Mnih et al., 2013; 2015)。在 DQN 之前，人们普遍认为使用大型非线性函数逼近器学习值函数是困难且不稳定的。DQN 之所以能够使用此类函数逼近器以稳定和稳健的方式学习值函数，得益于两项创新：1. 网络使用来自经验回放缓冲区（replay buffer）的样本进行离策略训练，以最小化样本间的相关性；2. 网络使用目标 Q 网络（target Q network）在时序差分备份（temporal difference backups）中提供一致的目标值。在本工作中，我们使用了同样的思想，以及批归一化（batch normalization）(Ioffe & Szegedy, 2015) 这一深度学习的最新进展。

为了评估我们的方法，我们构建了一系列具有挑战性的物理控制问题，涉及复杂的多关节运动、不稳定且丰富的接触动力学以及步态行为。其中包括经典问题如倒立摆摆起问题，以及许多新的任务域。机器人控制的一个长期挑战是直接从原始感知输入（如视频）学习动作策略。因此，我们在模拟器中放置了一个固定视角的相机，并尝试使用低维观测（如关节角度）和直接从像素完成所有任务。

我们的无模型方法称为深度 DPG（Deep DPG, DDPG），能够使用低维观测（如笛卡尔坐标或关节角度）在所有任务上学习具有竞争力的策略，且使用相同的超参数和网络结构。在许多情况下，我们也能够直接从像素学习良好的策略，同样保持超参数和网络结构不变。

该方法的一个关键特征是其简单性：它只需要一个直接的演员-评论家架构和学习算法，几乎没有"活动部件"，使其易于实现并扩展到更困难的问题和更大的网络。对于物理控制问题，我们将结果与一个规划器 (Tassa et al., 2012) 计算的基线进行比较，该规划器可以完全访问底层的模拟动力学及其导数（见补充信息）。有趣的是，DDPG 有时能找到超越规划器性能的策略，在某些情况下即使从像素学习也是如此（规划器总是在底层的低维状态空间上进行规划）。

---

## 2 背景

我们考虑一个标准的强化学习设定，由一个智能体（agent）在离散时间步与环境 $E$ 交互。在每个时间步 $t$，智能体接收观测 $x_t$，执行动作 $a_t$ 并获得标量奖励 $r_t$。在本文考虑的所有环境中，动作为实值 $a_t \in \mathbb{R}^N$。一般来说，环境可能是部分可观的，因此可能需要观测-动作对的完整历史 $s_t = (x_1, a_1, \ldots, a_{t-1}, x_t)$ 来描述状态。这里我们假设环境是完全可观的，因此 $s_t = x_t$。

智能体的行为由策略（policy）$\pi$ 定义，策略将状态映射到动作的概率分布 $\pi: \mathcal{S} \to \mathcal{P}(\mathcal{A})$。环境 $E$ 也可能是随机的。我们将其建模为一个马尔可夫决策过程（Markov decision process），具有状态空间 $\mathcal{S}$、动作空间 $\mathcal{A} = \mathbb{R}^N$、初始状态分布 $p(s_1)$、转移动力学 $p(s_{t+1}|s_t, a_t)$ 和奖励函数 $r(s_t, a_t)$。

从某一状态开始的回报（return）定义为折扣未来奖励之和：

$$R_t = \sum_{i=t}^{T} \gamma^{(i-t)} r(s_i, a_i)$$

其中折扣因子 $\gamma \in [0, 1]$。注意回报取决于所选择的动作，因此取决于策略 $\pi$，并且可能是随机的。强化学习的目标是学习一个使从初始分布开始的期望回报最大化的策略：

$$J = \mathbb{E}_{r_i, s_i \sim E, a_i \sim \pi} [R_1]$$

我们将策略 $\pi$ 的折扣状态访问分布记为 $\rho^\pi$。

动作值函数（action-value function）在许多强化学习算法中被使用。它描述了在状态 $s_t$ 中执行动作 $a_t$ 后，遵循策略 $\pi$ 的期望回报：

$$Q^\pi(s_t, a_t) = \mathbb{E}_{r_{i \geq t}, s_{i>t} \sim E, a_{i>t} \sim \pi} [R_t | s_t, a_t] \tag{1}$$

许多强化学习方法利用称为贝尔曼方程（Bellman equation）的递归关系：

$$Q^\pi(s_t, a_t) = \mathbb{E}_{r_t, s_{t+1} \sim E} \left[ r(s_t, a_t) + \gamma \mathbb{E}_{a_{t+1} \sim \pi} [Q^\pi(s_{t+1}, a_{t+1})] \right] \tag{2}$$

如果目标策略是确定性的，我们可以将其描述为函数 $\mu: \mathcal{S} \to \mathcal{A}$，从而避免内部期望：

$$Q^\mu(s_t, a_t) = \mathbb{E}_{r_t, s_{t+1} \sim E} [r(s_t, a_t) + \gamma Q^\mu(s_{t+1}, \mu(s_{t+1}))] \tag{3}$$

该期望仅依赖于环境。这意味着可以使用从不同随机行为策略 $\beta$ 生成的转移来离策略地学习 $Q^\mu$。

Q 学习 (Watkins & Dayan, 1992) 是一种常用的离策略算法，使用贪心策略 $\mu(s) = \arg\max_a Q(s, a)$。我们考虑由参数 $\theta^Q$ 参数化的函数逼近器，通过最小化以下损失进行优化：

$$L(\theta^Q) = \mathbb{E}_{s_t \sim \rho^\beta, a_t \sim \beta, r_t \sim E} \left[ (Q(s_t, a_t | \theta^Q) - y_t)^2 \right] \tag{4}$$

其中

$$y_t = r(s_t, a_t) + \gamma Q(s_{t+1}, \mu(s_{t+1}) | \theta^Q) \tag{5}$$

虽然 $y_t$ 也依赖于 $\theta^Q$，但通常忽略这一点。

使用大型非线性函数逼近器学习值函数或动作值函数在过去经常被避免，因为理论上无法提供性能保证，实践中学习也往往不稳定。近年来 (Mnih et al., 2013; 2015) 改进了 Q 学习算法，使其能够有效利用大型神经网络作为函数逼近器。他们的算法能够从像素学习玩 Atari 游戏。为了扩展 Q 学习，他们引入了两个重要改变：使用经验回放缓冲区，以及用于计算 $y_t$ 的独立目标网络。我们在 DDPG 的框架中采用了这些技术，并在下一节中解释其实现。

---

## 3 算法

在连续动作空间中直接应用 Q 学习是不可行的，因为在连续空间中寻找贪心策略需要在每个时间步对 $a_t$ 进行优化；对于大型无约束函数逼近器和非平凡动作空间，这种优化过于缓慢。因此，我们采用了基于 DPG 算法 (Silver et al., 2014) 的演员-评论家方法。

DPG 算法维护一个参数化的演员函数 $\mu(s|\theta^\mu)$，通过确定性地将状态映射到特定动作来指定当前策略。评论家 $Q(s, a)$ 如 Q 学习一样使用贝尔曼方程进行学习。演员通过对期望回报 $J$ 关于演员参数应用链式法则进行更新：

$$\nabla_{\theta^\mu} J \approx \mathbb{E}_{s_t \sim \rho^\beta} \left[ \nabla_{\theta^\mu} Q(s, a | \theta^Q) |_{s=s_t, a=\mu(s_t|\theta^\mu)} \right]$$

$$= \mathbb{E}_{s_t \sim \rho^\beta} \left[ \nabla_a Q(s, a | \theta^Q) |_{s=s_t, a=\mu(s_t)} \nabla_{\theta^\mu} \mu(s|\theta^\mu)|_{s=s_t} \right] \tag{6}$$

Silver et al. (2014) 证明这就是策略梯度（policy gradient），即策略性能的梯度。

与 Q 学习一样，引入非线性函数逼近器意味着不再保证收敛。然而这类逼近器对于在大状态空间上学习和泛化似乎是必不可少的。NFQCA (Hafner & Riedmiller, 2011) 使用与 DPG 相同的更新规则，但使用神经网络函数逼近器，并采用批量学习（batch learning）以保持稳定性，这对大型网络是不可行的。NFQCA 的小批量版本不在每次更新时重置策略（扩展到大型网络所需），等价于原始的 DPG。我们的贡献在于受 DQN 成功的启发，对 DPG 进行修改，使其能够使用神经网络函数逼近器在大状态和动作空间中在线学习。我们将我们的算法称为深度 DPG（DDPG，算法 1）。

使用神经网络进行强化学习时的一个挑战是，大多数优化算法假设样本是独立同分布的。显然，当样本是通过在环境中顺序探索生成时，这一假设不再成立。此外，为了有效利用硬件优化，在小批量（mini-batches）中学习而非在线学习是必要的。

与 DQN 一样，我们使用经验回放缓冲区来解决这些问题。回放缓冲区是一个有限大小的缓存 $\mathcal{R}$。根据探索策略从环境中采样转移，将元组 $(s_t, a_t, r_t, s_{t+1})$ 存储在回放缓冲区中。当缓冲区满时，丢弃最旧的样本。在每个时间步，从缓冲区中均匀采样一个小批量来更新演员和评论家。由于 DDPG 是离策略算法，回放缓冲区可以很大，使算法能够从一组不相关的转移中学习。

直接用神经网络实现 Q 学习（公式 4）在许多环境中被证明是不稳定的。由于正在更新的网络 $Q(s, a|\theta^Q)$ 同时也用于计算目标值（公式 5），Q 更新容易发散。我们的解决方案类似于 (Mnih et al., 2013) 中使用的目标网络，但针对演员-评论家进行了修改，并使用"软"目标更新（soft target updates），而非直接复制权重。我们创建演员和评论家网络的副本 $Q'(s, a|\theta^{Q'})$ 和 $\mu'(s|\theta^{\mu'})$，用于计算目标值。然后通过让目标网络的权重缓慢跟踪已学习的网络来更新：

$$\theta' \leftarrow \tau\theta + (1 - \tau)\theta'$$

其中 $\tau \ll 1$。这意味着目标值被约束为缓慢变化，极大地提高了学习的稳定性。这一简单改变使学习动作值函数这一相对不稳定的问题更接近于监督学习——一个已有稳健解决方案的问题。我们发现同时拥有目标 $\mu'$ 和 $Q'$ 对于获得稳定的目标 $y_i$ 以一致地训练评论家而不发散是必要的。这可能会减慢学习速度，因为目标网络延迟了值估计的传播。然而在实践中，我们发现学习稳定性的收益远大于这一代价。

当从低维特征向量观测学习时，观测的不同分量可能具有不同的物理单位（例如位置与速度），且其范围可能在不同环境中有所不同。这可能使网络难以有效学习，并难以找到跨不同状态值尺度的环境通用的超参数。

解决此问题的一种方法是手动缩放特征使其在不同环境和单位中处于相似范围。我们通过采用深度学习中称为批归一化（batch normalization）的技术 (Ioffe & Szegedy, 2015) 来解决这一问题。该技术在小批量中对每个维度进行归一化，使其具有单位均值和方差。此外，它维护均值和方差的运行平均值，用于测试时的归一化（在我们的情况中即探索或评估期间）。在深度网络中，它通过确保每层接收白化输入来最小化训练过程中的协变量偏移。在低维情况下，我们在状态输入以及 $\mu$ 网络的所有层和 $Q$ 网络动作输入之前的所有层上使用了批归一化（网络细节见补充材料）。通过批归一化，我们能够在具有不同类型单位的许多不同任务上有效学习，无需手动确保单位在设定范围内。

在连续动作空间中学习的一个主要挑战是探索（exploration）。离策略算法如 DDPG 的一个优势是我们可以独立于学习算法来处理探索问题。我们通过向演员策略添加从噪声过程 $\mathcal{N}$ 采样的噪声来构造探索策略 $\mu'$：

$$\mu'(s_t) = \mu(s_t|\theta_t^\mu) + \mathcal{N} \tag{7}$$

$\mathcal{N}$ 可以根据环境选择。如补充材料中详述的，我们使用 Ornstein-Uhlenbeck 过程 (Uhlenbeck & Ornstein, 1930) 来生成时间相关的探索噪声，以在具有惯性的物理控制问题中实现高效探索（类似的自相关噪声使用见 (Wawrzynski, 2015)）。

### 算法 1：DDPG 算法

> 随机初始化评论家网络 $Q(s, a|\theta^Q)$ 和演员 $\mu(s|\theta^\mu)$，权重分别为 $\theta^Q$ 和 $\theta^\mu$
>
> 初始化目标网络 $Q'$ 和 $\mu'$，权重 $\theta^{Q'} \leftarrow \theta^Q$，$\theta^{\mu'} \leftarrow \theta^\mu$
>
> 初始化经验回放缓冲区 $\mathcal{R}$
>
> **for** episode = 1, M **do**
>
> > 初始化用于动作探索的随机过程 $\mathcal{N}$
> >
> > 接收初始观测状态 $s_1$
> >
> > **for** t = 1, T **do**
> >
> > > 根据当前策略和探索噪声选择动作 $a_t = \mu(s_t|\theta^\mu) + \mathcal{N}_t$
> > >
> > > 执行动作 $a_t$，观测奖励 $r_t$ 和新状态 $s_{t+1}$
> > >
> > > 将转移 $(s_t, a_t, r_t, s_{t+1})$ 存入 $\mathcal{R}$
> > >
> > > 从 $\mathcal{R}$ 中随机采样 $N$ 个转移 $(s_i, a_i, r_i, s_{i+1})$ 的小批量
> > >
> > > 设置 $y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$
> > >
> > > 通过最小化损失更新评论家：$L = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i|\theta^Q))^2$
> > >
> > > 使用采样策略梯度更新演员策略：
> > >
> > > $\nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a|\theta^Q)|_{s=s_i, a=\mu(s_i)} \nabla_{\theta^\mu} \mu(s|\theta^\mu)|_{s_i}$
> > >
> > > 更新目标网络：
> > >
> > > $\theta^{Q'} \leftarrow \tau\theta^Q + (1 - \tau)\theta^{Q'}$
> > >
> > > $\theta^{\mu'} \leftarrow \tau\theta^\mu + (1 - \tau)\theta^{\mu'}$
> >
> > **end for**
>
> **end for**

---

## 4 实验结果

我们构建了不同难度级别的模拟物理环境来测试我们的算法。这些环境包括经典的强化学习环境如倒立摆，以及困难的高维任务如抓取器、涉及接触的冰球击打（canada）和猎豹（cheetah）等运动任务 (Wawrzynski, 2009)。在除猎豹外的所有域中，动作为施加在驱动关节上的力矩。这些环境使用 MuJoCo (Todorov et al., 2012) 进行模拟。

> **图 1**：我们尝试用 DDPG 解决的部分环境示例截图。从左到右依次为：倒立摆摆起任务、到达任务、抓取并移动任务、冰球击打任务、单足平衡任务、两个运动任务和 Torcs（驾驶模拟器）。我们使用低维特征向量和高维像素输入来处理所有任务。

在所有任务中，我们使用低维状态描述（如关节角度和位置）和环境的高维渲染图两种方式进行实验。与 DQN (Mnih et al., 2013; 2015) 一样，为了使高维环境中的问题近似完全可观，我们使用了动作重复（action repeats）。对于智能体的每个时间步，我们进行 3 个模拟时间步，每次重复智能体的动作并渲染。因此报告给智能体的观测包含 9 个特征图（3 次渲染各自的 RGB），使智能体能够利用帧间差异推断速度。帧被下采样至 64x64 像素，8 位 RGB 值被转换为缩放至 [0, 1] 的浮点数。

我们在训练期间定期评估策略（不使用探索噪声）。

> **图 2**：使用 DPG 变体在部分域上的性能曲线：带批归一化的原始 DPG 算法（即小批量 NFQCA）（浅灰色），带目标网络（深灰色），带目标网络和批归一化（绿色），带目标网络的纯像素输入（蓝色）。目标网络至关重要。
>
> *图中展示了 Cart、Pendulum Swing-up、Cartpole Swing-up、Fixed Reacher、Blockworld、Gripper、Puck Shooting、Monoped Balancing、Moving Gripper、Cheetah 等环境在训练步数（百万步）下的归一化奖励曲线。*

为了在所有任务上表现良好，目标网络和批归一化这两项改进都是必要的。特别是，不使用目标网络（如 DPG 原始工作中那样）的学习在许多环境中表现很差。

令人惊讶的是，在一些较简单的任务中，从像素学习策略与使用低维状态描述符一样快。这可能是因为动作重复使问题变得更简单。也可能是卷积层提供了一种容易分离的状态空间表示，便于高层快速学习。

> **表 1**：在所有环境中训练最多 250 万步后的性能。我们报告了平均值和最佳观测值（5 次运行中）。除 Torcs 外的所有分数均进行了归一化，使得随机智能体得分为 0，规划算法得分为 1；对于 Torcs，我们展示原始奖励分数。包括低维（lowd）和高维像素（pix）版本的 DDPG 结果，以及带经验回放缓冲区和批归一化的原始 DPG 算法（cntrl）的比较结果。
>
> *表格包含 blockworld1、blockworld3da、canada、canada2d、cart、cartpole、cartpoleBalance 等 20 余个环境的 $R_{av,lowd}$、$R_{best,lowd}$、$R_{av,pix}$、$R_{best,pix}$、$R_{av,cntrl}$、$R_{best,cntrl}$ 数值结果。DDPG 在许多任务中能够学到良好策略，在某些情况下甚至超越 iLQG 规划器。*

学习准确的值估计可能具有挑战性。例如 Q 学习容易高估值 (Hasselt, 2010)。我们通过比较训练后 Q 估计的值与测试回合中观察到的真实回报，对 DDPG 的估计进行了实证检验。

> **图 3**：估计 Q 值与测试回合中观察到的回报的密度图（5 次重复）。在简单域如摆和倒立摆中，Q 值相当准确。在更复杂的任务中，Q 估计不太准确，但仍可用于学习有竞争力的策略。虚线表示单位线。

为了展示方法的通用性，我们还包括了 Torcs——一个赛车游戏，动作为加速、刹车和转向。Torcs 此前在其他策略学习方法中被用作测试平台 (Koutnik et al., 2014b)。我们使用与物理任务完全相同的网络架构和学习算法超参数，但由于涉及的时间尺度非常不同而修改了探索的噪声过程。在低维和像素输入两种情况下，一些重复实验能够学习到合理的策略并完成赛道一圈，但其他重复实验未能学到有意义的策略。

---

## 5 相关工作

DPG 原始论文使用瓷砖编码（tile-coding）和线性函数逼近器在玩具问题上评估了该算法。它展示了离策略 DPG 相比在策略和离策略随机演员-评论家方法的数据效率优势。它还解决了一个更具挑战性的任务，其中多关节章鱼臂需要用肢体的任意部分击中目标。然而该论文未展示将该方法扩展到大规模高维观测空间，而我们在此实现了这一点。

人们通常假设标准策略搜索方法（如本工作中探索的方法）太脆弱而无法扩展到困难问题 (Levine et al., 2015)。标准策略搜索被认为是困难的，因为它同时处理复杂的环境动力学和复杂的策略。事实上，大多数使用演员-评论家和策略优化方法的过去工作在扩展到更具挑战性的问题时都遇到了困难 (Deisenroth et al., 2013)。这通常是由于学习不稳定性——问题上的进展被后续学习更新破坏，或者学习太慢而不实用。

近期的无模型策略搜索工作表明它可能不像先前假设的那么脆弱。Wawrzynski (2009); Wawrzynski & Tanwani (2013) 在带有经验回放缓冲区的演员-评论家框架中训练了随机策略。与我们的工作同期，Balduzzi & Ghifary (2015) 通过一个明确学习 $\partial Q / \partial a$ 的"偏差器"（deviator）网络扩展了 DPG 算法，但他们仅在两个低维域上训练。Heess et al. (2015) 引入了 SVG(0)，同样使用 Q 评论家但学习随机策略。DPG 可以被视为 SVG(0) 的确定性极限。我们描述的用于扩展 DPG 的技术也适用于随机策略，通过使用重参数化技巧（reparameterization trick）(Heess et al., 2015; Schulman et al., 2015a)。

另一种方法是信赖域策略优化（Trust Region Policy Optimization, TRPO）(Schulman et al., 2015b)，直接构建随机神经网络策略，而不将问题分解为最优控制和监督阶段。该方法通过对策略参数进行精心选择的更新来实现回报的近单调提升，约束更新以防止新策略偏离现有策略太远。这种方法不需要学习动作值函数，因此似乎数据效率显著较低。

为了应对演员-评论家方法的挑战，近期的引导策略搜索（Guided Policy Search, GPS）算法 (Levine et al., 2015) 将问题分解为三个相对容易解决的阶段：首先使用全状态观测在一个或多个标称轨迹周围创建动力学的局部线性近似；然后使用最优控制沿这些轨迹找到局部线性最优策略；最后使用监督学习训练一个复杂的非线性策略（如深度神经网络）来复现优化轨迹的状态到动作映射。

该方法有几个优点，包括数据效率，已成功应用于使用视觉的多种真实世界机器人操控任务。在这些任务中，GPS 使用与我们类似的卷积策略网络，但有两个显著区别：1. 它使用空间 softmax 将视觉特征的维度降低为每个特征图的单个 $(x, y)$ 坐标；2. 策略还在网络的第一个全连接层接收关于机器人配置的直接低维状态信息。这两者都可能增强算法的能力和数据效率，且可以很容易地在 DDPG 框架中利用。

PILCO (Deisenroth & Rasmussen, 2011) 使用高斯过程学习非参数的概率动力学模型。利用该学习模型，PILCO 计算解析策略梯度，在多个控制问题中实现了令人印象深刻的数据效率。然而由于高计算需求，PILCO "对高维问题不实用" (Wahlstrom et al., 2015)。深度函数逼近器似乎是将强化学习扩展到大规模高维域最有前景的方法。

Wahlstrom et al. (2015) 使用深度动态模型网络结合模型预测控制（model predictive control）从像素输入解决摆摆起任务。他们训练了一个可微的前向模型，并将目标状态编码到学习的潜在空间中。他们在学习模型上使用模型预测控制来寻找到达目标的策略。然而该方法仅适用于能够向算法演示目标状态的域。

近年来，进化方法也被用于从像素学习 Torcs 的竞争性策略，使用压缩权重参数化 (Koutnik et al., 2014a) 或无监督学习 (Koutnik et al., 2014b) 来降低进化权重的维度。这些方法泛化到其他问题的能力尚不明确。

---

## 6 结论

本工作结合了深度学习和强化学习的最新进展，提出了一种能够在连续动作空间的多种域中稳健解决具有挑战性问题的算法，即使使用原始像素作为观测也是如此。与大多数强化学习算法一样，非线性函数逼近器的使用消除了任何收敛保证；然而我们的实验结果表明，在不同环境之间无需任何修改即可实现稳定学习。

有趣的是，我们所有实验使用的经验步数远少于 DQN 在 Atari 域中找到解所需的步数。我们研究的几乎所有问题都在 250 万步经验内解决（通常远少于此），这比 DQN 获得良好 Atari 解所需的步数少了 20 倍。这表明给予更多的模拟时间，DDPG 可能解决比本文考虑的更困难的问题。

我们的方法仍存在一些局限性。最显著的是，与大多数无模型强化学习方法一样，DDPG 需要大量的训练回合才能找到解。然而我们相信，一种稳健的无模型方法可能是更大系统的重要组成部分，这些更大系统可以攻克这些局限性 (Glascher et al., 2010)。

---

## 参考文献

Balduzzi, David and Ghifary, Muhammad. Compatible value gradients for reinforcement learning of continuous deep policies. arXiv preprint arXiv:1509.03005, 2015.

Deisenroth, Marc and Rasmussen, Carl E. Pilco: A model-based and data-efficient approach to policy search. In Proceedings of the 28th International Conference on machine learning (ICML-11), pp. 465–472, 2011.

Deisenroth, Marc Peter, Neumann, Gerhard, Peters, Jan, et al. A survey on policy search for robotics. Foundations and Trends in Robotics, 2(1-2):1–142, 2013.

Gläscher, Jan, Daw, Nathaniel, Dayan, Peter, and O'Doherty, John P. States versus rewards: dissociable neural prediction error signals underlying model-based and model-free reinforcement learning. Neuron, 66(4):585–595, 2010.

Glorot, Xavier, Bordes, Antoine, and Bengio, Yoshua. Deep sparse rectifier networks. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics. JMLR W&CP Volume, volume 15, pp. 315–323, 2011.

Hafner, Roland and Riedmiller, Martin. Reinforcement learning in feedback control. Machine learning, 84(1-2):137–169, 2011.

Hasselt, Hado V. Double q-learning. In Advances in Neural Information Processing Systems, pp. 2613–2621, 2010.

Heess, N., Hunt, J. J, Lillicrap, T. P, and Silver, D. Memory-based control with recurrent neural networks. NIPS Deep Reinforcement Learning Workshop (arXiv:1512.04455), 2015.

Heess, Nicolas, Wayne, Gregory, Silver, David, Lillicrap, Tim, Erez, Tom, and Tassa, Yuval. Learning continuous control policies by stochastic value gradients. In Advances in Neural Information Processing Systems, pp. 2926–2934, 2015.

Ioffe, Sergey and Szegedy, Christian. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

Kingma, Diederik and Ba, Jimmy. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Koutník, Jan, Schmidhuber, Jürgen, and Gomez, Faustino. Evolving deep unsupervised convolutional networks for vision-based reinforcement learning. In Proceedings of the 2014 conference on Genetic and evolutionary computation, pp. 541–548. ACM, 2014a.

Koutník, Jan, Schmidhuber, Jürgen, and Gomez, Faustino. Online evolution of deep convolutional network for vision-based reinforcement learning. In From Animals to Animats 13, pp. 260–269. Springer, 2014b.

Krizhevsky, Alex, Sutskever, Ilya, and Hinton, Geoffrey E. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097–1105, 2012.

Levine, Sergey, Finn, Chelsea, Darrell, Trevor, and Abbeel, Pieter. End-to-end training of deep visuomotor policies. arXiv preprint arXiv:1504.00702, 2015.

Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Graves, Alex, Antonoglou, Ioannis, Wierstra, Daan, and Riedmiller, Martin. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel, Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski, Georg, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Prokhorov, Danil V, Wunsch, Donald C, et al. Adaptive critic designs. Neural Networks, IEEE Transactions on, 8(5):997–1007, 1997.

Schulman, John, Heess, Nicolas, Weber, Theophane, and Abbeel, Pieter. Gradient estimation using stochastic computation graphs. In Advances in Neural Information Processing Systems, pp. 3510–3522, 2015a.

Schulman, John, Levine, Sergey, Moritz, Philipp, Jordan, Michael I, and Abbeel, Pieter. Trust region policy optimization. arXiv preprint arXiv:1502.05477, 2015b.

Silver, David, Lever, Guy, Heess, Nicolas, Degris, Thomas, Wierstra, Daan, and Riedmiller, Martin. Deterministic policy gradient algorithms. In ICML, 2014.

Tassa, Yuval, Erez, Tom, and Todorov, Emanuel. Synthesis and stabilization of complex behaviors through online trajectory optimization. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 4906–4913. IEEE, 2012.

Todorov, Emanuel and Li, Weiwei. A generalized iterative lqg method for locally-optimal feedback control of constrained nonlinear stochastic systems. In American Control Conference, 2005. Proceedings of the 2005, pp. 300–306. IEEE, 2005.

Todorov, Emanuel, Erez, Tom, and Tassa, Yuval. Mujoco: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026–5033. IEEE, 2012.

Uhlenbeck, George E and Ornstein, Leonard S. On the theory of the brownian motion. Physical review, 36(5):823, 1930.

Wahlström, Niklas, Schön, Thomas B, and Deisenroth, Marc Peter. From pixels to torques: Policy learning with deep dynamical models. arXiv preprint arXiv:1502.02251, 2015.

Watkins, Christopher JCH and Dayan, Peter. Q-learning. Machine learning, 8(3-4):279–292, 1992.

Wawrzyński, Paweł. Real-time reinforcement learning by sequential actor–critics and experience replay. Neural Networks, 22(10):1484–1497, 2009.

Wawrzyński, Paweł. Control policy with autocorrelated noise in reinforcement learning for robotics. International Journal of Machine Learning and Computing, 5:91–95, 2015.

Wawrzyński, Paweł and Tanwani, Ajay Kumar. Autonomous reinforcement learning with experience replay. Neural Networks, 41:156–167, 2013.

---

## 补充信息

### 7 实验细节

我们使用 Adam (Kingma & Ba, 2014) 学习神经网络参数，演员和评论家的学习率分别为 $10^{-4}$ 和 $10^{-3}$。对于 $Q$，我们包含了 $10^{-2}$ 的 $L_2$ 权重衰减，折扣因子 $\gamma = 0.99$。软目标更新使用 $\tau = 0.001$。神经网络在所有隐藏层中使用修正线性非线性（rectified non-linearity）(Glorot et al., 2011)。演员的最终输出层为 tanh 层，用于限制动作范围。低维网络有 2 个隐藏层，分别包含 400 和 300 个单元（约 130,000 个参数）。动作直到 $Q$ 的第 2 个隐藏层才被引入。从像素学习时，我们使用 3 个卷积层（无池化），每层 32 个滤波器，后接两个包含 200 个单元的全连接层（约 430,000 个参数）。演员和评论家的最终层权重和偏置从均匀分布 $[-3 \times 10^{-3}, 3 \times 10^{-3}]$（低维情况）和 $[-3 \times 10^{-4}, 3 \times 10^{-4}]$（像素情况）中初始化，以确保策略和值估计的初始输出接近零。其他层从均匀分布 $[-1/\sqrt{f}, 1/\sqrt{f}]$ 初始化，其中 $f$ 为该层的扇入（fan-in）。动作直到全连接层才被引入。低维问题使用 64 的小批量大小训练，像素问题使用 16。经验回放缓冲区大小为 $10^6$。

对于探索噪声过程，我们使用时间相关噪声以在具有动量的物理环境中良好地探索。我们使用了 Ornstein-Uhlenbeck 过程 (Uhlenbeck & Ornstein, 1930)，参数 $\theta = 0.15$，$\sigma = 0.2$。Ornstein-Uhlenbeck 过程建模带有摩擦的布朗粒子的速度，产生以 0 为中心的时间相关值。

### 8 规划算法

我们的规划器实现为模型预测控制器 (Tassa et al., 2012)：在每个时间步，我们从系统的真实状态开始运行一次轨迹优化迭代（使用 iLQG (Todorov & Li, 2005)）。每次轨迹优化的规划范围在 250ms 到 600ms 之间，随着世界模拟的展开，规划范围会后退，这与模型预测控制一致。

iLQG 迭代从先前策略的初始展开开始，确定标称轨迹。我们使用模拟动力学的重复采样来近似轨迹每一步周围动力学的线性展开以及代价函数的二次展开。我们使用这一系列局部线性二次模型沿标称轨迹向后时间积分值函数。这一反向传播产生对动作序列的假设修改，将降低总代价。我们通过前向积分动力学（前向传播）在动作序列空间中沿此方向执行无导数线搜索，选择最佳轨迹。我们存储此动作序列以热启动下一次 iLQG 迭代，并在模拟器中执行第一个动作。这产生一个新状态，用作下一次轨迹优化迭代的初始状态。

### 9 环境细节

#### 9.1 Torcs 环境

对于 Torcs 环境，我们使用的奖励函数在每一步为汽车沿赛道方向的速度分量提供正奖励，碰撞时给予 -1 惩罚。如果在 500 帧后沿赛道没有取得进展，回合终止。

#### 9.2 MuJoCo 环境

对于物理控制任务，我们使用在每一步提供反馈的奖励函数。在所有任务中，奖励包含一个小的动作代价。对于所有具有静态目标状态的任务（如摆摆起和到达），我们提供基于到目标状态距离的平滑变化奖励，在某些情况下当处于目标状态小半径内时还有额外正奖励。对于抓取和操控任务，我们使用包含鼓励向有效载荷移动的项和鼓励将有效载荷移向目标的项的奖励。在运动任务中，我们奖励前进动作并惩罚硬碰撞以鼓励平滑而非跳跃的步态 (Schulman et al., 2015b)。此外，对于跌倒使用负奖励和提前终止，跌倒由高度和躯干角度的简单阈值确定（在 walker2d 的情况下）。

> **表 2**：MuJoCo 任务的维度：底层物理模型维度 dim(s)、动作维度 dim(a) 和观测维度 dim(o)。
>
> *表格列出了所有环境的状态、动作和观测维度。例如 cart 为 dim(s)=2, dim(a)=1, dim(o)=3；cheetah 为 dim(s)=18, dim(a)=6, dim(o)=17 等。*

**各环境简要描述：**

- **blockworld1**：智能体需使用约束在二维平面上的带抓取器手臂抓取下落方块，并将其抬起至固定目标位置。
- **blockworld3da**：智能体需使用 7 自由度类人手臂和简单抓取器抓取方块并抬升至固定目标位置。
- **canada**：智能体需使用 7 自由度手臂和冰球棍状附件将球击向目标。
- **canada2d**：智能体需使用带冰球棍状附件的手臂将随机初始位置的球击向随机目标位置。
- **cart**：智能体需将简单质量块移至 0 位。质量块每次试验从随机位置和随机速度开始。
- **cartpole**：经典倒立摆摆起任务。智能体仅通过对小车施力来平衡连接在小车上的杆。杆在每回合开始时倒挂。
- **cartpoleBalance**：经典倒立摆平衡任务。杆在每回合开始时处于直立位置。
- **cartpoleParallelDouble**：经典倒立摆变体。两根杆均连接在小车上，应尽可能保持直立。
- **cartpoleSerialDouble**：经典倒立摆变体。两根杆串联连接，应尽可能保持直立。
- **cartpoleSerialTriple**：经典倒立摆变体。三根杆串联连接，应尽可能保持直立。
- **cheetah**：智能体应以约束在平面上的猎豹体型尽快向前移动。该环境基于 Wawrzynski (2009) 的工作。
- **fixedReacher**：智能体需将 3 自由度手臂移至固定目标位置。
- **fixedReacherDouble**：智能体需将 2 自由度手臂移至固定目标位置。
- **fixedReacherSingle**：智能体需将简单 1 自由度手臂移至固定目标位置。
- **gripper**：智能体需使用带抓取器附件的手臂抓取物体并将其移至固定目标。
- **gripperRandom**：与 gripper 相同，但手臂、物体和目标位置随机初始化。
- **hardCheetah**：与 cheetah 类似但更困难，移除了模型中的稳定关节刚度。
- **hopper**：智能体需平衡多自由度单足以防止跌倒。
- **hyq**：智能体需防止基于 HyQ 机器人的四足模型跌倒。
- **movingGripper**：智能体需使用安装在可移动平台上的带抓取器手臂抓取物体并移至固定目标。
- **movingGripperRandom**：与 movingGripper 相同，但物体位置、目标位置和手臂状态随机初始化。
- **pendulum**：经典摆摆起问题。摆应被带至直立位置并保持平衡。力矩限制防止智能体直接摆起。
- **reacher3daFixedTarget**：智能体需将 7 自由度类人手臂移至固定目标位置。
- **reacher3daRandomTarget**：智能体需将 7 自由度类人手臂从随机起始位置移至随机目标位置。
- **reacher**：智能体需将 3 自由度手臂从随机起始位置移至随机目标位置。
- **reacherSingle**：智能体需将简单 1 自由度手臂从随机起始位置移至随机目标位置。
- **reacherObstacle**：智能体需将 5 自由度手臂绕过障碍物移至随机目标位置。
- **walker2d**：智能体应以约束在平面上的双足行走器尽快向前移动，同时不跌倒或躯干前后倾斜过多。
