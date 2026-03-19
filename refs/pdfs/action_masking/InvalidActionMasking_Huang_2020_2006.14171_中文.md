# 深入探究策略梯度算法中的无效动作掩码

**Shengyi Huang**
College of Computing & Informatics, Drexel University, Philadelphia, PA 19104
sh3397@drexel.edu

**Santiago Ontanon** *
College of Computing & Informatics, Drexel University, Philadelphia, PA 19104
so367@drexel.edu

*（现就职于 Google）*

---

## 摘要

近年来，深度强化学习（Deep Reinforcement Learning, DRL）算法在许多具有挑战性的策略游戏中取得了最先进的性能。由于这些游戏具有复杂的规则，从完整的离散动作空间中采样的动作通常是无效的。在策略梯度（Policy Gradient）算法中，处理这一问题的常用方法是"掩盖"无效动作，仅从有效动作集合中进行采样。然而，该过程的内在机制仍然缺乏深入研究。本文表明，无效动作掩码（Invalid Action Masking）的标准工作机制对应于有效的策略梯度更新。更有趣的是，它通过在动作概率分布的计算过程中施加一个状态相关的可微函数来实现。此外，我们展示了该技术对策略梯度算法性能的关键重要性。具体而言，实验表明无效动作掩码在无效动作空间较大时具有良好的可扩展性，而常见的对无效动作施加负奖励的方法则会失效。最后，我们通过评估不同的动作掩码方案提供了更深入的见解，例如在使用掩码训练完成后移除掩码。

---

## 1 引言

深度强化学习算法已经在即时战略（Real-time Strategy, RTS）游戏 [21, 20] 和多人在线竞技（Multiplayer Online Battle Arena, MOBA）游戏 [1, 23] 等具有挑战性的领域中培养出了最先进的游戏智能体。由于这些游戏具有复杂的规则，不同状态的离散动作空间通常具有不同的大小。即一个状态可能有 5 个有效动作，而另一个状态可能有 7 个有效动作。为了将这些游戏形式化为具有统一动作集的标准强化学习问题，先前的工作将这些离散动作空间合并为一个包含所有状态可用动作的完整离散动作空间 [21, 1, 23]。虽然这种完整的离散动作空间使得应用 DRL 算法变得更加容易，但一个问题是，从该完整离散动作空间中采样的动作对于某些游戏状态可能是无效的，这些动作将不得不被丢弃。更糟糕的是，某些游戏具有极其庞大的完整离散动作空间，从中采样的动作通常是无效的。例如，Dota 2 的完整离散动作空间有 1,837,080 个维度 [1]，采样的动作可能是在金币不足时购买物品。为了避免在完整离散动作空间中反复采样无效动作，近期工作将策略梯度算法与一种称为无效动作掩码的技术结合使用，该技术"掩盖"无效动作，然后仅从有效动作中采样 [21, 1, 23]。然而，据我们所知，无效动作掩码的理论基础尚未被研究，其经验效果也缺乏充分的调查。

本文深入探究无效动作掩码，指出无效动作掩码产生的梯度对应于有效的策略梯度。更有趣的是，我们表明无效动作掩码实际上可以被视为在动作概率分布计算过程中施加一个状态相关的可微函数，以产生行为策略（Behavior Policy）。接下来，我们设计实验比较无效动作掩码与无效动作惩罚（Invalid Action Penalty）的性能——后者是一种常见方法，对无效动作给予负奖励，使智能体通过不执行任何无效动作来最大化奖励。实验表明，当无效动作空间增大时，无效动作掩码具有良好的可扩展性，智能体能够完成期望的任务，而无效动作惩罚甚至难以探索到最初的奖励。然后，我们设计实验回答两个问题：(1) 如果在智能体使用掩码训练后移除无效动作掩码，会发生什么？(2) 当我们通过朴素方式实现无效动作掩码——即从掩码后的动作概率分布中采样动作，但使用未掩码的动作概率分布更新策略梯度——智能体的性能如何？最后，我们在 GitHub 上公开了源代码以确保可复现性。

---

## 2 背景

我们考虑马尔可夫决策过程（Markov Decision Process, MDP）中的强化学习问题，记为 $(S, A, P, \rho_0, r, \gamma, T)$，其中 $S$ 是状态空间，$A$ 是离散动作空间，$P: S \times A \times S \to [0,1]$ 是状态转移概率，$\rho_0: S \to [0,1]$ 是初始状态分布，$r: S \times A \to \mathbb{R}$ 是奖励函数，$\gamma$ 是折扣因子，$T$ 是最大回合长度。一个随机策略 $\pi_\theta: S \times A \to [0,1]$，由参数向量 $\theta$ 参数化，为给定状态下的动作分配概率值。目标是最大化期望折扣回报：

$$J = \mathbb{E}_\tau \left[ \sum_{t=0}^{T-1} \gamma^t r_t \right]$$

其中 $\tau$ 是轨迹 $(s_0, a_0, r_0, s_1, \ldots, s_{T-1}, a_{T-1}, r_{T-1})$，且 $s_0 \sim \rho_0$，$s_t \sim P(\cdot|s_{t-1}, a_{t-1})$，$a_t \sim \pi_\theta(\cdot|s_t)$，$r_t = r(s_t, a_t)$。

记号 $P(\cdot|s_{t-1}, a_{t-1})$ 表示给定先前状态 $s_{t-1}$ 和动作 $a_{t-1}$ 后的状态转移分布，$s_t \sim P(\cdot|s_{t-1}, a_{t-1})$ 表示在时刻 $t$ 访问的状态 $s_t$ 从 $P(\cdot|s_{t-1}, a_{t-1})$ 中采样。类似地，$\pi_\theta(\cdot|s_t)$ 表示给定状态 $s_t$ 后的动作分布，$a_t \sim \pi_\theta(\cdot|s_t)$ 表示时刻 $t$ 的动作 $a_t$ 从 $\pi_\theta(\cdot|s_t)$ 中采样。

### 策略梯度算法

策略梯度算法的核心思想是求得期望折扣回报关于策略参数 $\theta$ 的策略梯度 $\nabla_\theta J$。通过梯度上升 $\theta = \theta + \nabla_\theta J$ 来最大化期望折扣奖励。早期工作提出了以下对目标 $J$ 的策略梯度估计 [19, 18]：

$$g_{\text{policy}} = \mathbb{E}_\tau [\nabla_\theta \log \pi_\theta(a_\tau|s_\tau) G_\tau] = \mathbb{E}_\tau \left[ \nabla_\theta \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) G_t \right] \tag{1}$$

其中 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ 表示时刻 $t$ 之后的折扣回报。

---

## 3 无效动作掩码

无效动作掩码是一种常用技术，用于在大型离散动作空间中避免反复生成无效动作 [21, 1, 23]。据我们所知，目前没有文献对无效动作掩码的实现提供详细描述。现有工作 [21, 1] 似乎将无效动作掩码视为辅助细节，通常仅用几句话进行描述。此外，也没有文献提供理论论证来解释它为何能与策略梯度算法配合使用。本节中，我们研究无效动作掩码的实现方式，并证明它确实对应于有效的策略梯度更新 [19]。更有趣的是，我们表明它通过在动作概率分布的计算过程中施加一个状态相关的可微函数来实现。

首先，让我们了解策略梯度算法通常如何生成离散动作。大多数策略梯度算法使用神经网络来表示策略，神经网络通常输出未归一化的分数（logits），然后通过 softmax 运算或等价操作将其转换为动作概率分布——本文后续将基于此框架进行分析。为便于说明，考虑一个具有动作集 $A = \{a_0, a_1, a_2, a_3\}$ 和状态集 $S = \{s_0, s_1\}$ 的 MDP，其中在初始状态 $s_0$ 中执行一个动作后立即到达终止状态 $s_1$，奖励始终为 +1。进一步考虑由 $\theta = [l_0, l_1, l_2, l_3] = [1.0, 1.0, 1.0, 1.0]$ 参数化的策略 $\pi_\theta$，为简化示例，该策略直接产生 $\theta$ 作为输出 logits。则在 $s_0$ 中有：

$$\pi_\theta(\cdot|s_0) = [\pi_\theta(a_0|s_0), \pi_\theta(a_1|s_0), \pi_\theta(a_2|s_0), \pi_\theta(a_3|s_0)] = \text{softmax}([l_0, l_1, l_2, l_3]) \tag{2}$$
$$= [0.25, 0.25, 0.25, 0.25]$$

$$\pi_\theta(a_i|s_0) = \frac{\exp(l_i)}{\sum_j \exp(l_j)}$$

此时，常规策略梯度算法将从 $\pi_\theta(\cdot|s_0)$ 中采样一个动作。假设 $a_0$ 从 $\pi_\theta(\cdot|s_0)$ 中被采样，策略梯度可按如下方式计算：

$$g_{\text{policy}} = \mathbb{E}_\tau \left[ \nabla_\theta \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) G_t \right] = \nabla_\theta \log \pi_\theta(a_0|s_0) G_0 = [0.75, -0.25, -0.25, -0.25]$$

$$(\nabla_\theta \log \text{softmax}(\theta)_j)_i = \begin{cases} 1 - \frac{\exp(l_j)}{\sum_j \exp(l_j)} & \text{if } i = j \\ -\frac{\exp(l_j)}{\sum_j \exp(l_j)} & \text{otherwise} \end{cases}$$

现在假设 $a_2$ 对于状态 $s_0$ 是无效的，唯一有效的动作是 $a_0, a_1, a_3$。无效动作掩码通过"掩盖"对应无效动作的 logits 来避免采样无效动作。这通常通过将待掩盖动作的 logits 替换为一个很大的负数 $M$（如 $M = -1 \times 10^8$）来实现。我们用 $\text{inv}_s$ 表示此掩码过程，并可按如下方式计算重新归一化的概率分布 $\pi'_\theta(\cdot|s_0)$：

$$\pi'_\theta(\cdot|s_0) = \text{softmax}(\text{inv}_s([l_0, l_1, l_2, l_3])) \tag{3}$$
$$= \text{softmax}([l_0, l_1, M, l_3]) = [\pi'_\theta(a_0|s_0), \pi'_\theta(a_1|s_0), \epsilon, \pi'_\theta(a_3|s_0)] \tag{4}$$
$$= [0.33, 0.33, 0.0000, 0.33]$$

其中 $\epsilon$ 是被掩盖的无效动作的结果概率，应为一个很小的数。如果 $M$ 选取为足够大的负值，选择被掩盖的无效动作 $a_2$ 的概率将几乎为零。完成一个回合后，策略根据以下梯度进行更新，我们称之为无效动作策略梯度：

$$g_{\text{invalid action policy}} = \mathbb{E}_\tau \left[ \nabla_\theta \sum_{t=0}^{T-1} \log \pi'_\theta(a_t|s_t) G_t \right] \tag{5}$$
$$= \nabla_\theta \log \pi'_\theta(a_0|s_0) G_0 = [0.67, -0.33, 0.0000, -0.33]$$

此示例突显了无效动作掩码不仅仅是"重新归一化概率分布"；它实际上使得对应无效动作 logits 的梯度变为零。

### 3.1 无效动作掩码产生有效的策略梯度

动作选择过程受到一个看似外部于 $\pi_\theta$ 的过程影响，该过程计算掩码。因此，一个自然的问题是策略梯度定理 [19] 如何适用。事实上，我们的分析表明，无效动作掩码的过程可以被视为在 $\pi'_\theta$ 的计算中施加的一个状态相关的可微函数，因此 $g_{\text{invalid action policy}}$ 可以被视为 $\pi'_\theta$ 的策略梯度更新。

**命题 1.** $g_{\text{invalid action policy}}$ 是策略 $\pi'_\theta$ 的策略梯度。

**证明.** 令 $s \in S$ 为任意状态，将无效动作掩码过程视为一个可微函数 $\text{inv}_s$，施加于策略 $\pi_\theta$ 在给定状态 $s$ 下输出的 logits $l(s)$。则有：

$$\pi'_\theta(\cdot|s_t) = \text{softmax}(\text{inv}_s(l(s)))$$

$$\text{inv}_s(l(s))_i = \begin{cases} l_i & \text{if } a_i \text{ is valid in } s \\ M & \text{otherwise} \end{cases}$$

显然，$\text{inv}_{s_t}$ 对 logits 中的每个元素施加的要么是恒等函数，要么是常数函数。由于这两类函数均可微，所以 $\text{inv}_s$ 是可微的。因此，$\pi'_\theta$ 对其参数 $\theta$ 是可微的。即对所有 $a \in A$、$s \in S$，$\frac{\partial \pi'_\theta(a|s)}{\partial \theta}$ 存在，这满足策略梯度定理 [19] 的假设。因此，$g_{\text{invalid action policy}}$ 是策略 $\pi'_\theta$ 的策略梯度。 $\square$

虽然命题 1 表明无效动作掩码在理论上受策略梯度定理 [19] 的支持，但需注意 $\text{inv}_s$ 是一个状态相关的可微函数。即给定长度为 $|A|$ 的向量 $x$ 以及两个具有不同数量无效动作的状态 $s, s'$，有 $\text{inv}_s(x) \neq \text{inv}_{s'}(x)$。

---

## 4 实验设置

在本文的其余部分，我们提供一系列实验结果以展示无效动作掩码的实际意义。

### 4.1 评估环境

我们使用 $\mu$RTS 作为测试平台，这是一个最小化的即时战略游戏，保留了使 RTS 游戏从人工智能角度具有挑战性的核心特征：同时且持续的动作、大分支因子和实时决策。游戏截图见图 1。它是我们实验的理想测试平台，因为 $\mu$RTS 中的动作空间呈组合增长，无效动作的数量也相应增长。以下是实验环境的技术细节。

> **图 1**：左侧是 $\mu$RTS 的游戏截图。方形单位是"基地"（浅灰色，可生产工人）、"兵营"（深灰色，可生产军事单位）和"资源矿"（绿色，工人可从中提取资源以生产更多单位）；圆形单位是"工人"（小型，深灰色）和军事单位（大型，黄色或浅蓝色）。右侧是用于训练智能体采集资源的 10×10 地图。智能体控制左上角的单位，左下角的单位保持静止。

- **观测空间**：给定大小为 $h \times w$ 的地图，观测为形状 $(h, w, n_f)$ 的张量，其中 $n_f$ 是具有二值的特征平面数。本文使用 27 个特征平面（见附录表 3），类似于先前在 $\mu$RTS 中的工作 [17, 22, 8]。特征平面可以理解为多个独热编码特征的拼接。例如，如果有一个生命值为 1、未携带任何资源、属于玩家 1、当前未执行任何动作的工人，则独热编码特征如下：
  $$[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]$$
  该工人在地图位置上的 27 维特征平面值为：
  $$[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]$$

- **动作空间**：给定大小为 $h \times w$ 的地图，动作是一个 8 维离散值向量，如表 1 所示。动作空间设计类似于 Hausknecht 等人 [7] 的动作空间形式化方法。动作向量的第一个分量代表地图中要发出动作的单位，第二个是动作类型，其余分量代表不同动作类型可接受的参数。根据选择的动作类型，游戏引擎将使用相应参数执行动作。

> **表 1**：动作分量及其描述。
>
> | 动作分量 | 范围 | 描述 |
> |---------|------|------|
> | 源单位（Source Unit） | $[0, h \times w - 1]$ | 选择执行动作的单位位置 |
> | 动作类型（Action Type） | $[0, 5]$ | 空操作、移动、采集、返回、生产、攻击 |
> | 移动参数（Move Parameter） | $[0, 3]$ | 北、东、南、西 |
> | 采集参数（Harvest Parameter） | $[0, 3]$ | 北、东、南、西 |
> | 返回参数（Return Parameter） | $[0, 3]$ | 北、东、南、西 |
> | 生产方向参数（Produce Direction Parameter） | $[0, 3]$ | 北、东、南、西 |
> | 生产类型参数（Produce Type Parameter） | $[0, 5]$ | 资源、基地、兵营、工人、轻型、重型、远程 |
> | 攻击目标单位（Attack Target Unit） | $[0, h \times w - 1]$ | 将被攻击的单位位置 |

- **奖励**：我们评估智能体在为控制地图左上角单位的玩家 1 尽快采集资源的简单任务上的表现。工人采集一个资源时给予 +1 奖励，工人将资源返回基地时再给予 +1 奖励。

- **终止条件**：最大游戏长度设为 200 个时间步，但如果地图中所有资源先被采集完毕，游戏可提前终止。

注意，随着地图变大，无效动作空间会显著增大。这是因为动作空间中第一个和最后一个离散值的范围（对应源单位和攻击目标单位选择）随地图大小线性增长。在我们的实验中，通常只有两个单位可以被选为源单位（基地和工人）。虽然可以生产更多单位或建筑来选择，但生产行为对奖励没有贡献，因此通常不会被智能体学习到。注意源单位的范围在 $4 \times 4$ 和 $24 \times 24$ 地图中分别为 $4 \times 4 = 16$ 和 $24 \times 24 = 576$。在 $4 \times 4$ 地图中随机选择有效源单位的概率为 $2/16 = 0.125$，在 $24 \times 24$ 地图中为 $2/576 = 0.0034$。通过这样的动作空间，我们可以检验无效动作掩码的可扩展性。

### 4.2 训练算法

我们使用近端策略优化（Proximal Policy Optimization, PPO）[16] 作为训练智能体的 DRL 算法。实现细节和神经网络架构、超参数见附录 A。

### 4.3 处理无效动作的策略

为了检验无效动作掩码的经验重要性，我们比较以下四种处理无效动作的策略：

1. **无效动作惩罚（Invalid Action Penalty）**：每当智能体发出无效动作时，游戏环境向当前时间步产生的奖励添加一个非正奖励 $r_{\text{invalid}} \leq 0$。此技术在先前工作中很常见 [4]。我们分别实验 $r_{\text{invalid}} \in \{0, -0.01, -0.1, -1\}$ 以研究不同负奖励尺度的影响。

2. **无效动作掩码（Invalid Action Masking）**：在每个时间步 $t$，智能体在源单位和攻击目标单位特征上接收掩码，使得只有有效的单位可以被选择和作为目标。注意在我们的实验中，无效动作仍然可能被采样，因为智能体仍可能为当前动作类型选择不正确的参数。为简化起见，我们没有提供功能完备的无效动作掩码，因为源单位和攻击目标单位上的掩码已经显著缩小了动作空间。

3. **朴素无效动作掩码（Naive Invalid Action Masking）**：在每个时间步 $t$，智能体接收与无效动作掩码中描述的相同的源单位和攻击目标单位掩码。动作仍按公式 (4) 中计算的重新归一化概率进行采样——这确保不会采样到无效动作——但梯度根据公式 (2) 中计算的概率进行更新。我们称此实现为朴素无效动作掩码，因为其梯度没有将无效动作对应 logits 的梯度替换为零。

4. **移除掩码（Masking Removed）**：在每个时间步 $t$，智能体接收与无效动作掩码中描述的相同的源单位和攻击目标单位掩码，并以与无效动作掩码训练的智能体相同的方式进行训练。然而，我们随后在不提供掩码的情况下评估智能体。换言之，在此场景中，我们评估使用掩码训练但不使用掩码执行时会发生什么。

我们在 $4 \times 4$、$10 \times 10$、$16 \times 16$ 和 $24 \times 24$ 大小的地图上评估智能体的性能。所有地图中每个玩家有一个基地和一个工人，每个工人位于资源附近。

### 4.4 评估指标

我们使用以下指标来衡量实验中智能体的性能：

- $r_{\text{episode}}$：最近 10 个回合的平均回合奖励
- $a_{\text{null}}$：最近 10 个回合中选择无效源单位的平均动作数
- $a_{\text{busy}}$：最近 10 个回合中选择正在执行其他动作的"忙碌"源单位的平均动作数
- $a_{\text{owner}}$：最近 10 个回合中选择不属于玩家 1 的源单位的平均动作数
- $t_{\text{solve}}$：智能体最近 10 个回合的移动平均回合奖励超过 40 所需的总训练时间步占比
- $t_{\text{first}}$：智能体获得首个正奖励所需的总训练时间步占比

### 4.5 评估结果

结果报告在图 2 和表 2 中。以下是重要观察：

> **图 2**：(a) 和 (b) 展示了采用不同无效动作处理策略的智能体的学习曲线。x 轴为游戏步数，y 轴为平均回合奖励。(c) 和 (d) 的 x 轴为游戏步数，y 轴为 PPO 目标策略与当前策略之间的平均 KL 散度（Kullback-Leibler Divergence）。阴影区域表示 4 个随机种子数据的一个标准差。曲线经过平滑处理以提高可读性。其他地图的结果见附录图 3。

> **表 2**：4 个随机种子的平均结果。"-" 表示"不适用"。$r_{\text{episode}}$ 越高越好，$a_{\text{null}}$、$a_{\text{busy}}$、$a_{\text{owner}}$、$t_{\text{solve}}$ 和 $t_{\text{first}}$ 越低越好。（表中详细数据展示了四种策略在 4×4、10×10、16×16、24×24 地图上各指标的完整对比。）

**无效动作掩码具有良好的可扩展性。** 无效动作掩码在无效动作数量增加时表现出良好的可扩展性；$t_{\text{solve}}$ 大约为 12%，在不同地图大小之间非常相似。此外，无效动作掩码的 $t_{\text{first}}$ 不仅在所有实验中最低（仅占总时间步的约 0.05%–0.08%），而且在不同地图大小间保持一致。这意味着无论地图大小如何，智能体都能非常快速地找到第一个奖励。

**无效动作惩罚不可扩展。** 无效动作惩罚能在 $4 \times 4$ 地图中取得良好结果，但无法扩展到更大的地图。随着无效动作空间增大，它有时甚至难以发现第一个奖励。例如在 $10 \times 10$ 地图中，使用 $r_{\text{invalid}} = -0.01$ 的无效动作惩罚训练的智能体花费了整个训练时间的 3.43% 才发现第一个奖励，而使用无效动作掩码训练的智能体在所有地图中大约只需 0.06% 的时间。此外，超参数 $r_{\text{invalid}}$ 可能难以调节。虽然使用负的 $r_{\text{invalid}}$ 确实鼓励智能体不执行任何无效动作（如 $a_{\text{null}}$、$a_{\text{busy}}$、$a_{\text{owner}}$ 通常非常接近零），但设置 $r_{\text{invalid}} = -1$ 似乎有抑制智能体探索的副作用，因此在所有地图中始终表现最差。

**朴素无效动作掩码的 KL 爆炸。** 根据表 2，朴素无效动作掩码的 $r_{\text{episode}}$ 在几乎所有地图中都是最佳的。在 $4 \times 4$ 地图中，使用朴素无效动作掩码训练的智能体甚至学会了穿越到地图另一侧采集额外资源。然而，朴素无效动作掩码有两个主要问题：(1) 如图 2c、2d 所示，朴素无效动作掩码的 PPO 目标策略与当前策略之间的平均 KL 散度显著高于任何其他实验。由于策略在更新之间变化如此剧烈，朴素无效动作掩码的性能在处理更具挑战性的任务时可能会受到影响。(2) 如表 2 所示，朴素无效动作掩码的 $t_{\text{solve}}$ 更加不稳定且对地图大小敏感。例如在 $24 \times 24$ 地图中，使用朴素无效动作掩码训练的智能体需要整个训练时间的 49.14% 才能收敛。相比之下，使用无效动作掩码训练的智能体在所有地图中展示出一致的 $t_{\text{solve}} \approx 12\%$。

**移除掩码后仍在一定程度上保持行为。** 如图 2a、2b 所示，移除掩码后的智能体仍能在一定程度上良好运行。随着地图大小增大，其性能下降，并开始执行更多无效动作——最突出的是选择无效的源单位。尽管如此，其性能显著优于使用无效动作惩罚训练的智能体，即便前者在评估时不使用无效动作掩码。这表明使用无效动作掩码训练的智能体在无法再提供无效动作掩码时，仍能在一定程度上产生有用的行为。

---

## 5 相关工作

处理无效动作还有其他方法。Dulac-Arnold、Evans 等人 [5] 建议将离散动作空间嵌入连续动作空间，使用最近邻方法定位最近的有效动作。在自然语言游戏领域，其他人提出训练动作消除网络（Action Elimination Network, AEN）[24] 来缩减动作集。

避免执行无效动作的目的可以说是为了提高探索效率。一些相关性较低的工作通过将完整离散动作空间简化为更简单的动作空间来实现此目的。Kanervisto 等人 [10] 将这类工作描述为"动作空间塑形（Action Space Shaping）"，通常涉及：(1) 动作移除——如 Minecraft RL 环境移除如"潜行"等无用动作 [9]；(2) 连续动作空间的离散化——如著名的 CartPole-v0 环境将施加在小车上的连续力离散化 [2]。虽然精心塑形的动作空间可以帮助智能体高效探索并学习有用的策略，但动作空间塑形被证明可能难以调节，有时甚至对帮助智能体完成期望任务产生不利影响 [5]。

最后，Kanervisto 等人 [10] 和 Ye 等人 [23] 提供了消融研究以表明无效动作掩码对智能体性能可能很重要，但他们没有研究随着无效动作空间增大时无效动作掩码的经验效果——这正是本文所解决的问题。

---

## 6 结论

本文研究了无效动作掩码技术，这是策略梯度算法中常用的技术，用于避免执行无效动作，尤其是在动作空间较大的领域。我们的工作表明：(1) 无效动作掩码产生的梯度是有效的策略梯度；(2) 它通过在动作概率分布的计算过程中施加一个状态相关的可微函数来实现；(3) 无效动作掩码在无效动作空间增大时经验上具有良好的可扩展性——相比之下，当发出无效动作时给予负奖励的常见技术无法扩展，有时甚至在我们的环境中难以找到第一个奖励；(4) 使用无效动作掩码训练的智能体在移除掩码后仍能产生有用的行为。

对于未来工作，我们希望提供更好的理论框架来解释无效动作掩码的工作机制。特别是，我们计划研究使用动作掩码是否对学习算法的收敛保证有任何影响，并设计在训练期间可以利用掩码但不依赖掩码的方法，以便将其应用于训练时可以提供掩码但部署到现实世界时掩码不可用的应用领域。

---

## 参考文献

[1] Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemyslaw Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, Rafal Józefowicz, Scott Gray, Catherine Olsson, Jakub W. Pachocki, Michael Petrov, Henrique Pond'e de Oliveira Pinto, Jonathan Raiman, Tim Salimans, Jeremy Schlatter, Jonas Schneider, Szymon Sidor, Ilya Sutskever, Jie Tang, Filip Wolski, and Susan Zhang. Dota 2 with large scale deep reinforcement learning. ArXiv, abs/1912.06680, 2019.

[2] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

[3] Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, Yuhuai Wu, and Peter Zhokhov. Openai baselines. https://github.com/openai/baselines, 2017.

[4] Thomas G Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. Journal of artificial intelligence research, 13:227–303, 2000.

[5] Gabriel Dulac-Arnold, Richard Evans, Hado van Hasselt, Peter Sunehag, Timothy Lillicrap, Jonathan Hunt, Timothy Mann, Theophane Weber, Thomas Degris, and Ben Coppin. Deep reinforcement learning in large discrete action spaces. arXiv preprint arXiv:1512.07679, 2015.

[6] Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. Implementation matters in deep rl: A case study on ppo and trpo. In International Conference on Learning Representations, 2019.

[7] Matthew Hausknecht and Peter Stone. Deep reinforcement learning in parameterized action space. arXiv preprint arXiv:1511.04143, 2015.

[8] Shengyi Huang and Santiago Ontanon. Comparing observation and action representations for deep reinforcement learning in µrts. 2019.

[9] Matthew Johnson, Katja Hofmann, Tim Hutton, and David Bignell. The malmo platform for artificial intelligence experimentation. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI'16, page 4246–4247. AAAI Press, 2016.

[10] Anssi Kanervisto, Christian Scheller, and Ville Hautamäki. Action space shaping in deep reinforcement learning. arXiv preprint arXiv:2004.00980, 2020.

[11] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[12] Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86, 1951.

[13] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML-10), pages 807–814, 2010.

[14] Marc'Aurelio Ranzato, Fu Jie Huang, Y-Lan Boureau, and Yann LeCun. Unsupervised learning of invariant feature hierarchies with applications to object recognition. In 2007 IEEE conference on computer vision and pattern recognition, pages 1–8. IEEE, 2007.

[15] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

[16] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[17] Marius Stanescu, Nicolas A. Barriga, Andy Hess, and Michael Buro. Evaluating real-time strategy game states using convolutional neural networks. 09 2016.

[18] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

[19] Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems, pages 1057–1063, 2000.

[20] Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350–354, 2019.

[21] Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Küttler, John Agapiou, Julian Schrittwieser, et al. Starcraft ii: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782, 2017.

[22] Zuozhi Yang and Santiago Ontanon. Learning map-independent evaluation functions for real-time strategy games. 2018 IEEE Conference on Computational Intelligence and Games (CIG), pages 1–7, 2018.

[23] Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, et al. Mastering complex control in moba games with deep reinforcement learning. arXiv preprint arXiv:1912.09729, 2019.

[24] Tom Zahavy, Matan Haroush, Nadav Merlis, Daniel J Mankowitz, and Shie Mannor. Learn what not to learn: Action elimination with deep reinforcement learning. In Advances in Neural Information Processing Systems, pages 3562–3573, 2018.

---

## 附录

### A PPO 训练算法详细信息

我们用于训练智能体的 DRL 算法是近端策略优化（PPO）[16]，目前可用的最先进算法之一。关于我们的 PPO 实现有两个重要细节需要说明，这些在原始论文中未详细阐述。第一个细节涉及如何在 OpenAI Gym 环境 [2] 的 gym-microrts [8] 中定义的多离散（MultiDiscrete）动作空间中生成动作，第二个细节是关于用于增强性能的各种代码级优化。正如 Engstrom、Ilyas 等人 [6] 所指出的，这些代码级优化对 PPO 的性能可能至关重要。

#### A.1 多离散动作生成

要在 $\mu$RTS 中执行动作 $a_t$，根据表 1，我们必须选择源单位、动作类型及其相应的动作参数。因此，总共有 $hw \times 6 \times 4 \times 4 \times 4 \times 4 \times 6 \times hw = 9216(hw)^2$ 个可能的离散动作（包括无效动作），随着地图大小的增加呈指数增长。如果直接将 PPO 应用于该离散动作空间，为 $9216(hw)^2$ 个可能的动作生成分布将在计算上非常昂贵。为简化这一组合动作空间，openai/baselines [3] 库提出将该离散动作视为由若干较小的独立离散动作组成。即 $a_t$ 由以下较小的动作组成：

$$a_t^{\text{Source Unit}}, a_t^{\text{Action Type}}, a_t^{\text{Move Parameter}}, a_t^{\text{Harvest Parameter}}, a_t^{\text{Return Parameter}}, a_t^{\text{Produce Direction Parameter}}, a_t^{\text{Produce Type Parameter}}, a_t^{\text{Attack Target Unit}}$$

策略梯度按以下方式更新（为简化起见不考虑 PPO 的裁剪）：

$$\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t = \sum_{t=0}^{T-1} \nabla_\theta \left( \sum_{d \in D} \log \pi_\theta(a_t^d|s_t) \right) G_t = \sum_{t=0}^{T-1} \nabla_\theta \log \left( \prod_{d \in D} \pi_\theta(a_t^d|s_t) \right) G_t$$

其中 $D = \{\text{Source Unit, Action Type, Move Parameter, Harvest Parameter, Return Parameter, Produce Direction Parameter, Produce Type Parameter, Attack Target Unit}\}$。

在实现上，对于范围为 $[0, x-1]$ 的每个动作分量，生成相应形状为 $x$ 的 logits（我们称之为动作分量 logits），每个 $a_t^d$ 从该动作分量 logits 中采样。由于这一思想，算法现在只需生成 $hw + 6 + 4 + 4 + 4 + 4 + 6 + hw = 2hw + 36$ 个 logits，显著少于 $9216(hw)^2$。据我们所知，这种处理大型多离散动作空间的方法仅在 Kanervisto 等人 [10] 中被提及。

#### A.2 代码级优化

以下是本实验中使用的代码级优化列表：

1. **优势归一化（Normalization of Advantages）**：基于广义优势估计（GAE）计算优势后，通过减去均值并除以标准差对优势向量进行归一化。

2. **观测归一化（Normalization of Observation）**：观测在输入 PPO 智能体之前进行预处理。原始观测通过减去运行均值并除以方差进行归一化，然后裁剪到某个范围内，通常为 $[-10, 10]$。

3. **奖励缩放（Rewards Scaling）**：类似地，奖励通过除以折扣回报的运行方差进行预处理，然后裁剪到某个范围内，通常为 $[-10, 10]$。

4. **价值函数损失裁剪（Value Function Loss Clipping）**：openai/baselines 的 PPO 实现以类似于 PPO 裁剪替代目标的方式裁剪价值函数损失：
$$V_{\text{loss}} = \max \left[ (V_{\theta_t} - V_{\text{targ}})^2, \left( V_{\theta_{t-1}} + \text{clip}(V_{\theta_t} - V_{\theta_{t-1}}, -\varepsilon, \varepsilon) \right)^2 \right]$$
其中 $V_{\text{targ}}$ 通过将 $V_{\theta_{t-1}}$ 与广义优势估计 [15] 计算的优势 $A$ 相加得到。

5. **Adam 学习率退火（Adam Learning Rate Annealing）**：Adam [11] 优化器的学习率随着智能体训练的时间步增加而衰减。

6. **小批量更新（Mini-batch Updates）**：openai/baselines 的 PPO 实现使用小批量计算梯度并更新策略，而非使用全部批量数据。小批量采样方案确保每个转移仅被采样一次，且所有采样的转移确实用于网络更新。

7. **全局梯度裁剪（Global Gradient Clipping）**：在每个 epoch 的每次更新迭代中，策略和价值网络的梯度被裁剪，使得"全局 $\ell_2$ 范数"（即所有参数梯度拼接后的范数）不超过 0.5。

8. **权重正交初始化（Orthogonal Initialization of Weights）**：全连接层的权重和偏置使用不同缩放的正交初始化方案。在我们的实验中，由于历史原因，始终使用缩放因子 1。

### B $\mu$RTS 环境设置的补充细节

$\mu$RTS 中的每个动作需要一些内部游戏时间（以 tick 计量）来完成。gym-microrts [8] 将执行采集动作、返回动作和移动动作的时间设为 10 个游戏 tick。一旦向特定单位发出动作，该单位将被视为"忙碌"单位，在当前动作完成之前无法再执行任何动作。为了防止 DRL 算法反复向"忙碌"单位发出动作，gym-microrts 允许执行 9 帧的跳帧，使得从智能体的角度来看，一旦在当前观测下执行采集、返回或移动动作，这些动作将在下一个观测中完成。所有实验均使用此跳帧设置。

### C 可复现性

> **表 3**：特征图列表及其描述。
>
> | 特征 | 平面数 | 描述 |
> |------|--------|------|
> | 生命值（Hit Points） | 5 | 0, 1, 2, 3, >=4 |
> | 资源（Resources） | 5 | 0, 1, 2, 3, >=4 |
> | 所有者（Owner） | 3 | 玩家 1, -, 玩家 2 |
> | 单位类型（Unit Types） | 8 | -, 资源, 基地, 兵营, 工人, 轻型, 重型, 远程 |
> | 当前动作（Current Action） | 6 | -, 移动, 采集, 返回, 生产, 攻击 |

> **表 4**：实验参数及其值。
>
> | 参数名称 | 参数值 |
> |---------|--------|
> | 总时间步 | 500,000 |
> | $\gamma$（折扣因子） | 0.99 |
> | $\lambda$（GAE 参数） | 0.97 |
> | $\varepsilon$（PPO 裁剪系数） | 0.2 |
> | $\eta$（熵正则化系数） | 0.01 |
> | $\omega$（梯度范数阈值） | 0.5 |
> | $K$（每 epoch PPO 更新迭代次数） | 10 |
> | $\alpha_\pi$（策略学习率） | 0.0003 |
> | $\alpha_v$（价值函数学习率） | 0.0003 |

> **表 5**：神经网络架构。以 $24 \times 24$ 地图的架构为例进行详细说明。神经网络的输入为形状 $(24, 24, 27)$ 的张量。第一隐藏层对输入张量卷积 16 个 $3 \times 3$ 的滤波器（步幅 1），后接 $2 \times 2$ 最大池化层 [14]，并应用 ReLU 非线性激活 [13]。第二隐藏层类似地卷积 32 个 $2 \times 2$ 的滤波器（步幅 1），后接 $2 \times 2$ 最大池化层并应用 ReLU。最后的隐藏层是一个包含 128 个 ReLU 单元的全连接线性层。输出层是一个具有 $2hw + 36 = 1188$ 个输出的全连接线性层。
>
> 四种地图大小的具体架构：
> - **4x4**：Conv2d(27,16,kernel_size=2) -> MaxPool2d(1) -> ReLU -> Flatten -> Linear(144,128) -> ReLU -> Linear(128,68)
> - **10x10**：Conv2d(27,16,kernel_size=3) -> MaxPool2d(1) -> ReLU -> Conv2d(16,32,kernel_size=3) -> MaxPool2d(1) -> ReLU -> Flatten -> Linear(1152,128) -> ReLU -> Linear(128,236)
> - **16x16**：Conv2d(27,16,kernel_size=3) -> MaxPool2d(1) -> ReLU -> Conv2d(16,32,kernel_size=3) -> MaxPool2d(1) -> ReLU -> Flatten -> Linear(4608,128) -> ReLU -> Linear(128,548)
> - **24x24**：Conv2d(27,16,kernel_size=3,stride=1) -> MaxPool2d(2) -> ReLU -> Conv2d(16,32,kernel_size=2,stride=1) -> MaxPool2d(2) -> ReLU -> Flatten -> Linear(800,128) -> ReLU -> Linear(128,1188)

> **图 3**（附录）：左列展示了采用不同无效动作处理策略的智能体的学习曲线，x 轴为游戏步数，y 轴为平均回合奖励。右列的 x 轴为游戏步数，y 轴为 PPO 目标策略与当前策略之间的平均 KL 散度。阴影区域表示 4 个随机种子数据的一个标准差。曲线经过平滑处理以提高可读性。包含 4x4、10x10、16x16、24x24 四种地图大小的完整结果。
