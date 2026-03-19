# 优先经验回放

**发表于 ICLR 2016 会议论文**

**作者：** Tom Schaul, John Quan, Ioannis Antonoglou, David Silver

**单位：** Google DeepMind

---

## 摘要

经验回放（Experience Replay）使在线强化学习智能体能够记忆并重用过去的经验。在先前的工作中，经验转移（transition）是从回放记忆（replay memory）中均匀采样的。然而，这种方法只是以与最初经历时相同的频率重放转移，而不考虑其重要性。本文提出了一种经验优先化框架，通过更频繁地重放重要的转移来提高学习效率。我们将优先经验回放（Prioritized Experience Replay, PER）应用于深度 Q 网络（Deep Q-Network, DQN），这是一种在多款 Atari 游戏中达到人类水平表现的强化学习算法。带有优先经验回放的 DQN 达到了新的最优水平，在 49 款游戏中的 41 款上超越了使用均匀回放的 DQN。

---

## 1 引言

在线强化学习（Reinforcement Learning, RL）智能体在观察经验流的过程中增量更新其参数（策略、价值函数或模型的参数）。在最简单的形式中，它们在一次更新后立即丢弃传入数据。这带来两个问题：(a) 强相关的更新破坏了许多流行随机梯度算法的独立同分布（i.i.d.）假设；(b) 可能有用的稀有经验被快速遗忘。

经验回放（Lin, 1992）解决了上述两个问题：将经验存储在回放记忆中后，可以通过混合较新和较旧的经验来打破时间相关性，并且稀有经验可以被多次使用。这在深度 Q 网络（DQN）算法（Mnih et al., 2013; 2015）中得到了验证，该算法通过使用经验回放稳定了由深度神经网络表示的价值函数的训练。具体来说，DQN 使用了一个大的滑动窗口回放记忆，从中均匀随机采样，每个转移$^1$平均被重放八次。总体而言，经验回放可以减少所需的经验量，用更多的计算和更多的存储来替代——这些资源通常比 RL 智能体与环境的交互更廉价。

本文研究如何通过优先化重放哪些转移，使经验回放比均匀重放所有转移更加高效和有效。核心思想是：RL 智能体从某些转移中能学到的东西比从其他转移中更多。转移可能具有不同程度的"意外性"、冗余性或任务相关性。某些转移可能当前对智能体没有用，但随着智能体能力的提升可能变得有用（Schmidhuber, 1991）。经验回放使在线学习智能体摆脱了按原始经历顺序处理转移的限制。优先回放则进一步使智能体摆脱以相同频率考虑转移的限制。

具体而言，我们提出更频繁地重放具有高期望学习进步的转移，以时序差分误差（Temporal-Difference error, TD error）的大小作为度量。这种优先化可能导致多样性损失，我们通过随机优先化来缓解；同时引入偏差，我们通过重要性采样（Importance Sampling, IS）来校正。最终算法具有鲁棒性和可扩展性，我们在 Atari 2600 基准测试集上进行了验证，获得了更快的学习速度和最优性能。

> $^1$ 转移是 RL 中交互的原子单位，在本文中为元组 $(S_{t-1}, A_{t-1}, R_t, \gamma_t, S_t)$。选择这种形式是为了简单，但本文的大部分论点也适用于更粗粒度的经验分块方式，例如序列或回合。

---

## 2 背景

许多神经科学研究在啮齿动物的海马体中发现了经验回放的证据，表明先前经验的序列会在清醒休息或睡眠期间被重放。与奖励相关的序列似乎被更频繁地重放（Atherton et al., 2015; Olafsdottir et al., 2015; Foster & Wilson, 2006）。具有高 TD 误差的经验也似乎被更频繁地重放（Singer & Frank, 2009; McNamara et al., 2014）。

众所周知，值迭代（Value Iteration）等规划算法可以通过以适当顺序优先化更新来提高效率。优先化扫描（Prioritized Sweeping）（Moore & Atkeson, 1993; Andre et al., 1998）选择下一个要更新的状态，根据执行该更新后价值的变化量进行优先排序。TD 误差提供了度量这些优先级的一种方式（van Seijen & Sutton, 2013）。我们的方法使用类似的优先化方法，但用于无模型 RL 而非基于模型的规划。此外，我们使用随机优先化，这在从样本中学习函数近似器时更加鲁棒。

TD 误差也被用作确定资源聚焦位置的优先化机制，例如选择在哪里探索（White et al., 2014）或选择哪些特征（Geramifard et al., 2011; Sun et al., 2011）。

在监督学习中，当类别标识已知时，有许多处理不平衡数据集的技术，包括重采样、欠采样和过采样技术，可能与集成方法结合使用（综述见 Galar et al., 2012）。最近的一篇论文在深度 RL 经验回放的背景下引入了一种重采样形式（Narasimhan et al., 2015），该方法将经验分为两个桶——正奖励桶和负奖励桶——然后从每个桶中选取固定比例进行重放。这仅适用于具有自然"正/负"经验概念的领域（与我们的不同）。此外，Hinton (2007) 引入了一种基于误差的非均匀采样形式，结合重要性采样校正，在 MNIST 数字分类上获得了 3 倍加速。

已有多种使用深度强化学习玩 Atari 的方法被提出，包括深度 Q 网络 DQN（Mnih et al., 2013; 2015; Guo et al., 2014; Stadie et al., 2015; Nair et al., 2015; Bellemare et al., 2016），以及双重 DQN（Double DQN）算法（van Hasselt et al., 2016），后者是当时已发表的最优方法。与我们的工作同期，一种将优势函数与价值函数分离的架构创新（见 Wang et al., 2015 的共同提交）也在 Atari 基准上带来了实质性改进。

---

## 3 优先回放

使用回放记忆涉及两个层面的设计选择：存储哪些经验，以及重放哪些经验（及如何重放）。本文仅讨论后者：在回放记忆内容不受我们控制的前提下，如何最有效地利用回放记忆进行学习（另见第 6 节）。

### 3.1 一个启发性示例

为理解优先化的潜在收益，我们引入了一个人工的"盲崖行走"（Blind Cliffwalk）环境（如图 1 左所示），它体现了奖励稀少时探索的挑战。仅有 $n$ 个状态，环境需要指数级数量的随机步骤才能获得第一个非零奖励；确切地说，随机动作序列到达奖励的概率为 $2^{-n}$。此外，最相关的转移（来自稀有成功）隐藏在大量高度冗余的失败案例中（类似于双足机器人在学会走路之前反复摔倒）。

我们使用该示例突出两个智能体学习时间的差异。两个智能体都从同一个回放记忆中抽取转移进行 Q-learning 更新。第一个智能体均匀随机重放转移，第二个智能体调用一个预言机（oracle）来优先化转移。该预言机贪婪地选择在当前状态下最大程度减少全局损失的转移（回顾性地，在参数更新之后）。设置细节见附录 B.1。

> **图 1 描述：** 左：盲崖行走示例环境示意图。有两个动作——"正确"和"错误"，每当智能体选择"错误"动作时回合终止（虚线红箭头）。选择"正确"动作则通过 $n$ 个状态的序列前进（黑箭头），序列末端有一个最终奖励 1（绿箭头）；其他位置奖励为 0。表示方式使得泛化哪个动作是"正确的"不可能。右：学习价值函数所需的学习步数中位数，作为回放记忆中转移总数的函数。注意对数-对数坐标，突出了使用预言机回放（亮蓝色）与均匀回放（黑色）之间的指数级加速；浅色线为 10 次独立运行的最小/最大值。

图 1（右）表明，以良好的顺序选取转移可以比均匀选择带来指数级加速。这样的预言机当然是不现实的，但巨大的差距激励我们寻找在实践中优于均匀随机回放的方法。

### 3.2 基于 TD 误差的优先化

优先回放的核心组件是度量每个转移重要性的标准。一个理想化的标准是 RL 智能体在当前状态下从一个转移中可以学到的量，即期望学习进步。虽然这一度量不可直接获得，但一个合理的代理指标是转移的 TD 误差 $\delta$ 的绝对值，它表示该转移有多"令人意外"或出乎意料：具体来说，价值与其下一步自举估计之间的差距有多大（Andre et al., 1998）。这对于已经计算 TD 误差并按 $\delta$ 的比例更新参数的增量在线 RL 算法特别适用，如 SARSA 或 Q-learning。TD 误差在某些情况下也可能是较差的估计，例如当奖励有噪声时；替代方案的讨论见附录 A。

为展示基于 TD 误差优先回放的潜在有效性，我们在盲崖行走上比较均匀基线和预言机基线与"贪婪 TD 误差优先化"算法。该算法将每个转移最近一次遇到的 TD 误差存储在回放记忆中。具有最大绝对 TD 误差的转移被从记忆中重放。Q-learning 更新被应用于该转移，按 TD 误差的比例更新权重。新到达的转移没有已知的 TD 误差，因此我们将它们设为最大优先级，以保证所有经验至少被看到一次。图 2（左）表明，该算法显著减少了解决盲崖行走任务所需的努力$^2$。

**实现：** 为扩展到大的记忆大小 $N$，我们使用二叉堆（binary heap）数据结构作为优先队列，其中找到最大优先级转移的采样操作为 $O(1)$，更新优先级（用学习步骤后的新 TD 误差）为 $O(\log N)$。更多细节见附录 B.2.1。

> $^2$ 注意，使用贪婪优先化时需要对 Q 值进行随机（或乐观）初始化。如果用零初始化，无奖励的转移最初看起来误差为零，会被放在队列底部，直到其他转移的误差降到数值精度以下才会被重新访问。

### 3.3 随机优先化

然而，贪婪 TD 误差优先化存在若干问题。首先，为避免对整个回放记忆进行昂贵的遍历，TD 误差仅对被重放的转移进行更新。一个后果是，首次访问时 TD 误差较低的转移可能长时间不会被重放（在滑动窗口回放记忆中实际上意味着永远不会）。此外，它对噪声尖峰敏感（例如当奖励是随机的），这可能因自举而加剧——自举中近似误差表现为另一种噪声源。最后，贪婪优先化聚焦于经验的一小部分子集：误差缩减缓慢，特别是在使用函数近似时，这意味着初始高误差转移被频繁重放。这种多样性的缺乏使系统容易过拟合（overfitting）。

为克服这些问题，我们引入了一种在纯贪婪优先化和均匀随机采样之间插值的随机采样方法。我们确保采样概率关于转移优先级是单调的，同时保证即使是最低优先级的转移也有非零的采样概率。具体地，我们定义采样转移 $i$ 的概率为：

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha} \tag{1}$$

其中 $p_i > 0$ 是转移 $i$ 的优先级。指数 $\alpha$ 决定使用多少优先化，$\alpha = 0$ 对应均匀情况。

我们考虑的第一个变体是直接的**比例优先化**（proportional prioritization），其中 $p_i = |\delta_i| + \epsilon$，这里 $\epsilon$ 是一个小的正常数，防止误差为零的转移不再被重放的边界情况。第二个变体是间接的**基于排名的优先化**（rank-based prioritization），其中 $p_i = \frac{1}{\text{rank}(i)}$，$\text{rank}(i)$ 是当回放记忆按 $|\delta_i|$ 排序时转移 $i$ 的排名。在这种情况下，$P$ 变为指数为 $\alpha$ 的幂律分布。两种分布都关于 $|\delta|$ 单调，但后者可能更鲁棒，因为它对异常值不敏感。两种随机优先化变体在崖行走任务上都比均匀基线带来了大幅加速，如图 2（右）所示。

> **图 2 描述：** Q-learning 在盲崖行走示例上学习价值函数所需更新次数的中位数，作为转移总数的函数（其中只有一个是成功的并看到了非零奖励）。浅色线为 10 次随机初始化的最小/最大值。黑色为均匀随机回放，青色使用回顾预言机选择转移，红色和蓝色使用优先回放（分别为基于排名和比例方法）。结果相差多个数量级，因此需要对数-对数图。左：表格表示，贪婪优先化。右：线性函数近似，两种随机优先化变体。

**实现：** 为从分布 (1) 中高效采样，复杂度不能依赖于 $N$。对于基于排名的变体，我们可以用具有 $k$ 个等概率段的分段线性函数近似累积分布函数。段边界可以预计算（仅当 $N$ 或 $\alpha$ 变化时才改变）。在运行时，我们先采样一个段，然后在该段内的转移中均匀采样。这与基于小批量的学习算法配合特别好：选择 $k$ 为小批量大小，从每个段中恰好采样一个转移——这是一种分层采样（stratified sampling）形式，额外的好处是平衡小批量（总会有恰好一个高 $|\delta|$ 的转移、一个中等 $|\delta|$ 的等）。比例变体不同，也允许基于"求和树"（sum-tree）数据结构的高效实现（每个节点是其子节点之和，优先级为叶节点），可以高效更新和采样。更多细节见附录 B.2.1。

### 算法 1：带有比例优先化的 Double DQN

```
输入：小批量大小 k，步长 η，回放周期 K 和大小 N，指数 α 和 β，预算 T
初始化回放记忆 H = ∅，Δ = 0，p₁ = 1
观察 S₀ 并选择 A₀ ~ πθ(S₀)
for t = 1 to T do
    观察 Sₜ, Rₜ, γₜ
    以最大优先级 pₜ = max_{i<t} pᵢ 将转移 (Sₜ₋₁, Aₜ₋₁, Rₜ, γₜ, Sₜ) 存入 H
    if t ≡ 0 mod K then
        for j = 1 to k do
            采样转移 j ~ P(j) = pⱼ^α / Σᵢ pᵢ^α
            计算重要性采样权重 wⱼ = (N · P(j))^{-β} / max_i wᵢ
            计算 TD 误差 δⱼ = Rⱼ + γⱼ Q_target(Sⱼ, argmax_a Q(Sⱼ, a)) - Q(Sⱼ₋₁, Aⱼ₋₁)
            更新转移优先级 pⱼ ← |δⱼ|
            累积权重变化 Δ ← Δ + wⱼ · δⱼ · ∇θ Q(Sⱼ₋₁, Aⱼ₋₁)
        end for
        更新权重 θ ← θ + η · Δ，重置 Δ = 0
        不时将权重复制到目标网络 θ_target ← θ
    end if
    选择动作 Aₜ ~ πθ(Sₜ)
end for
```

### 3.4 偏差退火

使用随机更新估计期望值依赖于这些更新对应的分布与期望的分布相同。优先回放以不受控的方式改变了该分布，因此引入了偏差，改变了估计将收敛到的解（即使策略和状态分布是固定的）。我们可以使用重要性采样权重来校正这一偏差：

$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

当 $\beta = 1$ 时完全补偿非均匀概率 $P(i)$。这些权重可以通过使用 $w_i \delta_i$ 代替 $\delta_i$ 融入 Q-learning 更新（这是加权重要性采样，而非普通重要性采样，见 Mahmood et al., 2014）。出于稳定性考虑，我们始终通过 $1/\max_i w_i$ 归一化权重，使其仅向下缩放更新。

在典型的强化学习场景中，更新的无偏性在训练末期接近收敛时最为重要，因为在此之前过程本身就是高度非平稳的（由于策略、状态分布和自举目标的变化）；我们假设在此背景下可以忽略小的偏差（另见附录中图 12 关于 Atari 上完全 IS 校正的案例研究）。因此我们利用了随时间退火重要性采样校正量的灵活性，通过定义指数 $\beta$ 的调度，使其仅在学习结束时达到 1。在实践中，我们将 $\beta$ 从初始值 $\beta_0$ 线性退火到 1。注意，该超参数的选择与优先化指数 $\alpha$ 的选择相互作用；同时增大两者会更积极地优先化采样，同时更强烈地校正偏差。

重要性采样在与优先回放结合用于非线性函数近似（如深度神经网络）时还有另一个好处：此时大的更新步可能非常具有破坏性，因为梯度的一阶近似仅在局部可靠，必须通过较小的全局步长来防止。在我们的方法中，优先化确保高误差转移被多次看到，而 IS 校正降低了梯度大小（从而减小了参数空间中的有效步长），使算法能够跟随高度非线性优化景观的曲率，因为泰勒展开被不断重新近似。

我们将优先回放算法整合到完整的强化学习智能体中，基于最优的 Double DQN 算法。我们的主要修改是用我们的随机优先化和重要性采样方法替换 Double DQN 使用的均匀随机采样（见算法 1）。

---

## 4 Atari 实验

有了上述所有概念，我们现在研究带有这种优先化采样的回放能在多大程度上改善真实问题领域的性能。为此，我们选择了 Atari 基准集合（Bellemare et al., 2012）及其端到端视觉 RL 设置，因为它们流行且包含多样化的挑战，包括延迟信用分配、部分可观测性和困难的函数近似（Mnih et al., 2015; van Hasselt et al., 2016）。我们的假设是优先回放具有普遍用途，使得经验回放学习更高效，而无需仔细的问题特定超参数调优。

我们考虑两个使用均匀经验回放的基线算法：Nature 论文版本的 DQN 算法（Mnih et al., 2015），及其最近的扩展 Double DQN（van Hasselt et al., 2016），后者通过双重 Q-learning（van Hasselt, 2010）减少过估计偏差从而显著改进了最优水平。本文使用调优版本的 Double DQN 算法。对本文来说最相关的基线组件是回放机制：所有经历的转移存储在保留最近 $10^6$ 个转移的滑动窗口记忆中。算法处理从记忆中均匀采样的 32 个转移的小批量。每进入 4 个新转移做一次小批量更新，因此所有经验平均被重放 8 次。奖励和 TD 误差被裁剪到 $[-1, 1]$ 以保证稳定性。

我们使用与基线完全相同的神经网络架构、学习算法、回放记忆和评估设置（见附录 B.2）。唯一的区别是从回放记忆中采样转移的机制，现在根据算法 1 而非均匀采样。我们将基线与两种优先回放变体（基于排名和比例）进行比较。

与基线相比仅需一个超参数调整：鉴于优先回放更频繁地选取高误差转移，典型梯度大小更大，因此我们将步长 $\eta$ 比（Double）DQN 设置减小 4 倍。对于优先化引入的 $\alpha$ 和 $\beta_0$ 超参数，我们进行了粗网格搜索（在 8 款游戏子集上评估），发现最佳值为：基于排名变体 $\alpha = 0.7$, $\beta_0 = 0.5$；比例变体 $\alpha = 0.6$, $\beta_0 = 0.4$。这些选择在激进性和鲁棒性之间做了权衡，但通过减小 $\alpha$ 和/或增大 $\beta$ 可以容易地恢复到更接近基线的行为。

我们使用单一超参数设置在所有游戏上运行每个变体来产生结果，与基线相同。我们的主要评估指标是最佳策略的质量，用每回合平均得分衡量，给定从人类轨迹采样的起始状态（如 Nair et al., 2015 所引入，van Hasselt et al., 2016 所使用，这要求更多的鲁棒性和泛化能力，因为智能体不能依赖重复单一记忆轨迹）。这些结果汇总在表 1 和图 3 中，完整结果和原始分数可在附录表 7 和表 6 中找到。次要指标是学习速度，汇总在图 4 中，更详细的学习曲线在图 7 和图 8 中。

> **图 3 描述：** 在 57 款游戏（人类起点）上的归一化分数差异（随机与人类之间的差距为 100%），比较带和不带优先回放的 Double DQN（基于排名变体为红色，比例变体为蓝色），在大多数游戏上显示出显著改进。

**表 1：归一化分数汇总**

|  | DQN | Double DQN (调优) | 基线 | 基于排名 | 基线 | 基于排名 | 比例 |
|---|---|---|---|---|---|---|---|
| 中位数 | 48% | 106% | | | 111% | 113% | 128% |
| 均值 | 122% | 355% | | | 418% | 454% | 551% |
| > 基线 | - | 41 | | | - | 38 | 42 |
| > 人类 | 15 | 25 | | | 30 | 33 | 33 |
| 游戏数 | 49 | 49 | | | 57 | 57 | 57 |

我们发现，将优先回放加入 DQN 在 49 款游戏中的 41 款上带来了显著的分数提升，49 款游戏的中位归一化性能从 48% 提升到 106%。此外，优先经验回放带来的提升与引入双重 Q-learning 到 DQN 的提升是互补的：性能进一步提升，达到 Atari 基准上的新最优水平（见图 3）。与 Double DQN 相比，57 款游戏的中位性能从 111% 提升到 128%，均值性能从 418% 提升到 551%，首次将 River Raid、Seaquest 和 Surround 等游戏带到人类水平，并在其他游戏上取得了大幅跃进（如 Gopher、James Bond 007 或 Space Invaders）。注意均值性能不是非常可靠的指标，因为单个游戏（Video Pinball）有主导性贡献。优先回放在几乎所有游戏上都带来了性能提升，总体上学习速度提高了两倍（见图 4 和图 8）。

> **图 4 描述：** 学习速度汇总图。左：57 款游戏中至今达到的最大基线归一化分数的中位数。基于排名和比例优先化分别在总训练时间的 47% 和 38% 处达到等效点。右：类似于左图但使用均值而非最大值，基于排名和比例优先化分别在总训练时间的 41% 和 43% 处达到等效点。

---

## 5 讨论

在基于排名优先化和比例优先化的正面比较中，我们预期基于排名的变体更鲁棒，因为它不受异常值和误差大小的影响。此外，其重尾特性也保证了样本的多样性，从不同误差分区的分层采样将使总小批量梯度在整个训练过程中保持稳定的大小。另一方面，排名使算法对相对误差尺度视而不见，当误差分布中存在可利用的结构时（例如稀疏奖励场景），可能导致性能下降。也许令人惊讶的是，两种变体在实践中表现相似；我们怀疑这是由于 DQN 算法大量使用裁剪（对奖励和 TD 误差的裁剪），消除了异常值。监测若干游戏中 TD 误差分布随时间的变化（见附录图 10），发现随着学习的进展它接近重尾分布，但在不同游戏间仍然差异显著；这从经验上验证了方程 (1) 的形式。附录图 11 展示了这种分布如何与方程 (1) 相互作用以产生有效的回放概率。

在进行这一分析时，我们偶然发现了另一个现象（事后来看很明显）：一部分被访问的转移在从滑动窗口记忆中消失之前从未被重放过，更多的转移在遇到后很久才首次被重放。此外，均匀采样隐含地偏向于过时的转移——这些转移是由一个通常已经过数十万次更新的策略生成的。优先回放通过对未见过转移的奖励直接纠正了第一个问题，也倾向于帮助解决第二个问题，因为较新的转移往往有更大的误差——这是因为旧转移有更多机会被纠正，而新数据往往不太能被价值函数很好地预测。

我们假设深度神经网络以另一种有趣的方式与优先回放交互。当我们区分在给定表示下学习价值（即顶层）和学习改进的表示（即底层）时，表示良好的转移将快速降低其误差然后被大大减少重放，增加对其他表示较差的转移的学习关注，从而投入更多资源来区分混叠状态——如果观测和网络容量允许的话。

---

## 6 扩展

**优先化监督学习：** 优先回放在监督学习背景下的类比方法是从数据集中非均匀采样，每个样本使用基于其最近一次误差的优先级。这可以帮助聚焦于那些仍然可以从中学习的样本，将额外资源投入（困难的）边界案例，类似于提升方法（boosting）（Galar et al., 2012）。此外，如果数据集是不平衡的，我们假设稀有类别的样本将被不成比例地频繁采样，因为它们的误差缩减更慢，而从常见类别中选出的样本将是最接近决策边界的，产生类似于困难负样本挖掘（hard negative mining）（Felzenszwalb et al., 2008）的效果。

为检验这些直觉是否成立，我们在经典 MNIST 数字分类问题的类别不平衡变体上进行了初步实验（LeCun et al., 1998），在训练集中移除了数字 0-4 的 99% 的样本，同时保持测试/验证集不变。我们比较两种场景：在知情情况下，我们人为地对贫乏类别的误差重新加权（100 倍）；在不知情情况下，我们不提供任何关于测试分布将与训练分布不同的提示。见附录 B.3 了解卷积神经网络训练设置的细节。优先化采样（不知情，$\alpha = 1$, $\beta = 0$）超越了不知情均匀基线，在泛化性能方面接近知情均匀基线的表现（见图 5）；同样，优先化训练在学习速度方面也更快。

> **图 5 描述：** 在严重类别不平衡的 MNIST 上，分类误差作为监督学习更新次数的函数。优先化采样改善了性能，使测试集误差与知情方法接近（3 次随机初始化的中位数）。左：被错误分类的测试集样本数。右：测试集损失，突出了过拟合现象。

**离策略回放：** 离策略 RL 的两种标准方法是拒绝采样和使用重要性采样比率 $\rho$ 来校正转移在当前策略下的可能性。我们的方法包含这两种方法的类比：回放概率 $P$ 和 IS 校正 $w$。因此将其应用于离策略 RL 似乎是很自然的（如果转移在回放记忆中可用的话）。特别是，我们可以在比例变体中通过 $w = \rho$, $\alpha = 0$, $\beta = 1$ 恢复加权 IS，通过 $p = \min(1; \rho)$, $\alpha = 1$, $\beta = 0$ 恢复拒绝采样。我们的实验表明，中间变体（可能带有退火或排名）在实践中可能更有用——特别是当 IS 比率引入高方差时，即当目标策略在某些状态下与行为策略差异很大时。当然，离策略校正与我们基于期望学习进步的优先化是互补的，同一框架可以用于混合优先化，通过定义 $p = \rho \cdot |\delta|$ 或其他基于 $\rho$ 和 $\delta$ 的合理权衡。

**探索反馈：** 优先回放的一个有趣的副作用是，一个转移最终被重放的总次数 $M_i$ 差异很大，这大致指示了它对智能体有多大用处。这个潜在有价值的信号可以反馈给生成转移的探索策略。例如，我们可以在每个回合开始时从参数化分布中采样探索超参数（如随机动作比例 $\epsilon$、Boltzmann 温度或混入的内在奖励量），通过 $M_i$ 监控经验的有用性，并更新分布以生成更有用的经验。或者在并行系统（如 Gorila 智能体）（Nair et al., 2015）中，它可以在具有不同探索超参数的异构"演员"集合之间指导资源分配。

**优先化记忆：** 帮助确定重放哪些转移的考量也可能与确定存储哪些记忆以及何时擦除它们相关（例如当它们不太可能再被重放时）。对保留或擦除哪些记忆的显式控制可以帮助减少所需的总记忆大小，因为它减少了冗余（频繁访问的转移误差低，因此许多会被丢弃），同时自动调整已学内容（丢弃许多"简单"转移），并将记忆内容偏向误差仍然较高的地方。这是一个不平凡的方面，因为 DQN 的内存需求目前由回放记忆的大小主导，而不再是神经网络的大小。擦除比降低回放概率是更终极的决定，因此可能需要更强调多样性，例如通过跟踪每个转移的年龄并用它来调节优先级，以保留足够的旧经验来防止循环（与多智能体文献中的"名人堂"思想相关，Rosin & Belew, 1997）。优先级机制也足够灵活，允许整合来自其他来源的经验，如规划器或人类专家轨迹（Guo et al., 2014），因为知道来源可以用来调节每个转移的优先级。

---

## 7 结论

本文介绍了优先回放，一种能使从经验回放中学习更高效的方法。我们研究了几种变体，设计了扩展到大型回放记忆的实现，发现优先回放将学习速度提高了 2 倍并在 Atari 基准上达到了新的最优性能。我们还提出了进一步的变体和扩展，其中类别不平衡监督学习的应用前景看好。

---

## 致谢

我们感谢 DeepMind 同事们，特别是 Hado van Hasselt、Joseph Modayil、Nicolas Heess、Marc Bellemare、Razvan Pascanu、Dharshan Kumaran、Daan Wierstra、Arthur Guez、Charles Blundell、Alex Pritzel、Alex Graves、Balaji Lakshminarayanan、Ziyu Wang、Nando de Freitas、Remi Munos 和 Geoff Hinton 的深刻讨论和反馈。

---

## 附录 A 优先化变体

绝对 TD 误差只是理想优先级度量（期望学习进步）的一种可能的代理。虽然它捕捉了潜在改进的规模，但它忽略了奖励或转移中的固有随机性，以及部分可观测性或函数近似容量的可能限制；换言之，当存在不可学习的转移时它是有问题的。在这种情况下，其导数——可以通过当前 $|\delta|$ 与上次重放时 $|\delta|$ 之间的差来近似$^3$——可能更有用。然而，这一度量不太容易获得，并且受到中间重放内容的影响，增加了其方差。在初步实验中，我们发现它并未超越 $|\delta|$，但这可能更多地说明我们研究的（近确定性）环境类别，而非度量本身。

一个正交变体是考虑重放一个转移所引起的权重变化的范数——如果底层优化器采用自适应步长来降低高噪声方向的梯度（Schaul et al., 2013; Kingma & Ba, 2014），这可能是有效的，从而将区分可学习和不可学习转移的负担放在优化器上。

可以通过不同等地处理正 TD 误差和负 TD 误差来调节优先化；例如我们可以援引安娜·卡列尼娜原则（Anna Karenina principle）（Diamond, 1994），解释为一个转移可以有许多方式比预期差，但只有一种方式可以比预期好，从而引入不对称性，对等大的正 TD 误差比负 TD 误差给予更高优先级，因为前者更可能提供信息。这种回放频率的不对称性也在大鼠研究中观察到（Singer & Frank, 2009）。同样，我们对此类变体的初步实验结论不明确。

神经科学的证据表明，基于回合回报而非期望学习进步的优先化也可能有用（Atherton et al., 2015; Olafsdottir et al., 2015; Foster & Wilson, 2006）。对于这种情况，我们可以提升整个回合的回放概率而非单个转移，或者按观察到的剩余回报（甚至价值估计）提升单个转移。

对于保持足够多样性的问题（防止过拟合、过早收敛或表示退化），除了我们选择引入随机性之外还有替代方案，例如可以用观测空间中的新颖性度量来调节优先级。混合方法也是可行的：每个小批量中的一部分元素根据一种优先级度量采样，其余根据另一种采样，引入额外多样性。一个正交思路是提高长时间未重放转移的优先级，通过引入显式的过时奖励（staleness bonus），保证每个转移不时被重新访问，概率随其最近一次 TD 误差变得过时而增长。在线性增长的简单情况下，可以通过在任何更新时从新优先级中减去一个与全局步数成比例的量来实现，无需额外开销$^4$。

在使用价值函数自举的 RL 特定情况下，可以利用回放记忆的序列结构：一个导致大量学习（关于其输出状态）的转移有可能改变所有进入该状态的转移的自举目标，因此关于这些转移有更多可学习的内容。当然我们至少知道其中一个——即历史前驱转移——因此提升其优先级使其更可能很快被重新访问。类似于资格迹（eligibility traces），这让信息从未来结果向导致它的动作和状态的价值估计回溯传播。在实践中，我们将当前转移的 $|\delta|$ 加到前驱转移的优先级上，但仅当前驱转移不是终端转移时。这个想法与啮齿动物中观察到的"反向回放"（Foster & Wilson, 2006）以及优先化扫描的最近扩展（van Seijen & Sutton, 2013）相关。

> $^3$ 当然，更鲁棒的近似是所有遇到的 $\delta$ 值历史的函数。特别是，可以想象一种 RProp 风格的更新（Riedmiller, 1994），当符号匹配时增加优先级，当同一转移的连续误差符号不同时降低优先级。

> $^4$ 如果使用带策略迭代的自举，使得目标值来自单独的网络（如 DQN 的情况），那么在目标网络在外层迭代中更新时，优先级的不确定性会大幅增加。在这些时点，过时奖励按中间发生的单个（底层）更新次数的比例增加。

---

## 附录 B 实验细节

### B.1 盲崖行走

对于盲崖行走实验（第 3.1 节及后续），我们使用了直接的 Q-learning（Watkins & Dayan, 1992）设置。Q 值使用查表或线性函数近似器表示，两种情况下均为 $Q(s, a) := \theta^\top \phi(s, a)$。对于每个转移，我们使用以下公式计算其 TD 误差：

$$\delta_t := R_t + \gamma_t \max_a Q(S_t, a) - Q(S_{t-1}, A_{t-1}) \tag{2}$$

并使用随机梯度上升更新参数：

$$\theta \leftarrow \theta + \eta \cdot \delta_t \cdot \nabla_\theta Q(S_{t-1}, A_{t-1}) = \theta + \eta \cdot \delta_t \cdot \phi(S_{t-1}, A_{t-1}) \tag{3}$$

对于线性函数近似情况，我们使用非常简单的状态编码——独热向量（与查表相同），但连接了一个值为 1 的常数偏置特征。为使跨动作的泛化不可能，我们在每个状态交替哪个动作是"正确"的和"错误"的。所有元素初始化为接近零的小值，$\theta_i \sim \mathcal{N}(0, 0.1)$。

我们将问题规模（状态数 $n$）从 2 变化到 16。折扣因子设为 $\gamma = 1 - \frac{1}{n}$，使得价值大致在相同尺度上独立于 $n$。这允许我们在所有实验中使用固定步长 $\eta = \frac{1}{4}$。

回放记忆通过穷举执行所有 $2^n$ 种可能的动作序列直到终止（以随机顺序）来填充。这保证恰好一个序列成功并获得最终奖励，所有其他序列以零奖励失败。回放记忆包含所有相关经验（转移总数为 $2^{n+1} - 2$），频率与使用随机行为策略在线行动时遇到的一致。鉴于此，原则上可以通过增加计算量来学习直到收敛；这里收敛定义为 Q 值估计与真实值之间的均方误差（MSE）低于 $10^{-3}$。

### B.2 Atari 实验

#### B.2.1 实现细节

使用 $N = 10^6$ 个转移的回放记忆进行优先化引入了一些性能挑战。以下描述了我们为最小化额外运行时间和内存开销所做的工作。

**基于排名的优先化：** Atari 上的早期实验表明，维护一个不断变化 TD 误差的 $10^6$ 个转移的排序数据结构主导了运行时间。我们的最终解决方案是将转移存储在基于数组的二叉堆实现的优先队列中。然后直接将堆数组作为排序数组的近似使用，每 $10^6$ 步不频繁地排序一次以防止堆变得过于不平衡。这是二叉堆的非常规使用，但我们在较小环境上的测试表明，与使用完美排序数组相比，学习不受影响。这可能是因为最近一次的 TD 误差只是转移有用性的代理，以及我们使用随机优先化采样。运行时间的小改进来自避免过多重新计算采样分布的分区。我们对接近的 $N$ 值重用相同的分区，并不频繁地更新 $\alpha$ 和 $\beta$。我们基于排名优先化的最终实现仅增加了 2%-4% 的运行时间和可忽略的额外内存使用。

**比例优先化：** 这里使用的"求和树"数据结构在精神上与二叉堆的数组表示非常相似。然而，不是通常的堆属性，父节点的值是其子节点之和。叶节点存储转移优先级，内部节点是中间求和，父节点包含所有优先级的总和 $p_{\text{total}}$。这提供了一种高效的方式来计算优先级的累积和，允许 $O(\log N)$ 的更新和采样。要采样大小为 $k$ 的小批量，将区间 $[0, p_{\text{total}}]$ 等分为 $k$ 个范围。然后从每个范围中均匀采样一个值。最后从树中检索对应于每个采样值的转移。开销类似于基于排名的优先化。

如第 3.4 节所述，每当使用重要性采样时，所有权重 $w_i$ 被缩放使得 $\max_i w_i = 1$。我们发现这在实践中效果更好，因为它将所有权重保持在合理范围内，避免了极大更新的可能性。值得一提的是，这种归一化与 $\beta$ 的退火相互作用：随着 $\beta$ 趋近 1，归一化常数增长，以类似于退火步长 $\eta$ 的方式减少有效平均更新。

#### B.2.2 超参数

本文的基线是 DQN 和调优版 Double DQN。我们在 Atari 游戏子集上调优了超参数：Breakout、Pong、Ms. Pac-Man、Q*bert、Alien、Battlezone、Asterix。

**表 2：实验中考虑的超参数**

| 超参数 | 取值范围 |
|--------|----------|
| $\alpha$ | 0, 0.4, 0.5, 0.6, 0.7, 0.8 |
| $\beta$ | 0, 0.4, 0.5, 0.6, 1 |
| $\eta$ | $\eta_{\text{baseline}}$, $\eta_{\text{baseline}}/2$, $\eta_{\text{baseline}}/4$, $\eta_{\text{baseline}}/8$ |

其中 $\eta_{\text{baseline}} = 0.00025$。

**表 3：优先化 DQN 变体的选定超参数**

| 超参数 | DQN 基线 | DQN 基于排名 | DDQN 基线 | DDQN 基于排名 | DDQN 比例 |
|--------|----------|-------------|-----------|--------------|-----------|
| $\alpha$ (优先级) | 0 | $0.5 \to 0$ | 0 | 0.7 | 0.6 |
| $\beta$ (IS) | 0 | 0 | 0 | $0.5 \to 1$ | $0.4 \to 1$ |
| $\eta$ (步长) | 0.00025 | $\eta_{\text{baseline}}/4$ | 0.00025 | $\eta_{\text{baseline}}/4$ | $\eta_{\text{baseline}}/4$ |

箭头表示线性退火，极限值在训练结束时达到。注意基于排名变体以 DQN 为基线时是不带 IS 的早期版本。此时优先回放引入的偏差通过将 $\alpha$ 退火到零来校正。

#### B.2.3 评估

我们使用 van Hasselt et al. (2016) 描述的人类起点评估方法来评估智能体。人类起点评估使用从人类轨迹中随机采样的起始状态。智能体在训练期间定期进行的测试评估使用在每个回合开始时执行随机数量的空操作来随机化起始状态。人类起点评估在 100 次 30 分钟游戏时间的评估上取平均分。所有学习曲线图显示测试评估下的分数，使用相同的代码库和相同的随机种子初始化生成。

**表 4：评估方法比较**

| 评估方法 | 帧数 | 模拟器时间 | 评估次数 | 智能体起点 |
|----------|------|-----------|---------|-----------|
| 人类起点 | 108,000 | 30 分钟 | 100 | 人类起点 |
| 测试 | 500,000 | 139 分钟 | 1 | 最多 30 次随机空操作 |

归一化分数按如下计算（van Hasselt et al., 2016）：

$$\text{score}_{\text{normalized}} = \frac{\text{score}_{\text{agent}} - \text{score}_{\text{random}}}{|\text{score}_{\text{human}} - \text{score}_{\text{random}}|} \tag{4}$$

注意分母取绝对值。这仅影响 Video Pinball，其中随机分数高于人类分数。

**表 5：各智能体在各评估方法下 $\epsilon$-贪婪策略中使用的 $\epsilon$**

| 评估方法 | DQN 基线 | DQN 基于排名 | DDQN 基线 | DDQN 基于排名 | DDQN 比例 |
|----------|----------|-------------|-----------|--------------|-----------|
| 人类起点 | 0.05 | 0.01 | 0.001 | 0.001 | 0.001 |
| 测试 | 0.05 | 0.05 | 0.01 | 0.01 | 0.01 |

### B.3 类别不平衡 MNIST

#### B.3.1 数据集设置

在监督学习设置中，我们修改了 MNIST 以获得具有显著标签不平衡的新训练数据集。该新数据集通过考虑前 5 个数字（0, 1, 2, 3, 4）的样本子集和剩余 5 个标签（5, 6, 7, 8, 9）的所有样本获得。对于前 5 个数字中的每一个，我们随机采样了可用示例的 1%。在结果数据集中，所有 10 个不同类别都有示例，但高度不平衡——5-9 类的示例是 0-4 类的 100 倍。所有实验中使用原始 MNIST 测试数据集，未移除任何样本。

#### B.3.2 训练设置

实验中使用了类似 LeNet5（Lecun et al., 1998）架构的 4 层前馈神经网络。这是一个 2 层卷积神经网络后接 2 个全连接层。每个卷积层由纯卷积、ReLU 非线性和子采样最大池化操作组成。网络中的两个全连接层也由 ReLU 非线性分隔。最后一层是 softmax，用于获得可能标签上的归一化分布。完整架构见图 6，使用 Torch7 实现（Collobert et al., 2011）。模型使用无动量的随机梯度下降训练，小批量大小为 60。所有实验中考虑了 6 种不同的步长（0.3, 0.1, 0.03, 0.01, 0.003 和 0.001），每种情况选择导致最佳（平衡）验证性能的步长。使用负对数似然损失准则，实验了加权和未加权版本。在加权情况下，前 5 个数字（0-4）的示例损失被缩放 100 倍以适应上述训练集中的标签不平衡。

> **图 6 描述：** 优先化监督学习实验中使用的前馈网络架构示意图。

---

## 附录图表描述

> **图 7 描述：** Atari 基准套件全部 57 款游戏的学习曲线（原始分数）。Double DQN 均匀基线（黑色）、基于排名优先回放（红色）、比例优先化（蓝色），以及原始 DQN（灰色）。每条曲线对应一次 2 亿唯一帧的训练运行，使用测试评估，经 10 点移动平均平滑。

> **图 8 描述：** 基于排名（红色）和比例（蓝色）优先化与均匀 Double DQN 基线（黑色）在选定游戏上的详细学习曲线。实线为中位数分数，阴影区域为 8 次随机初始化的四分位距。虚线绿色为人类分数。运行间变异性显著，但最终达到的分数和学习速度存在显著差异。

> **图 9 描述：** 在 49 款游戏（人类起点）上的归一化分数差异，比较带和不带基于排名优先回放的 DQN，在许多游戏上显示出显著改进。

> **图 10 描述：** 回放记忆中所有转移的最近一次绝对 TD 误差的可视化（排序后），用于选定 Atari 游戏。线条按训练期间的时间用颜色编码（冷色为开始，暖色为结束）。观察到在某些游戏中开始时相当尖峰但很快变得分散，大致遵循重尾分布。这一现象在基于排名优先回放（上）和均匀回放（下）中都发生，但优先回放更快。

> **图 11 描述：** 有效回放概率作为绝对 TD 误差的函数，用于基于排名优先回放变体在训练开始附近。展示了方程 (1) 在 $\alpha = 0.7$ 时的实际效果，与均匀基线（虚线水平线）对比。效果不规则，但对所选游戏定性上相似。

> **图 12 描述：** 重要性采样的效果：这些学习曲线展示了基于排名优先化如何受到完全重要性采样校正（$\beta = 1$，橙色）的影响，与均匀基线（黑色，$\alpha = 0$）和纯粹的未校正优先回放（紫色，$\beta = 0$）在几款选定游戏上比较。阴影区域为四分位距。完全 IS 校正的步长与均匀回放相同。未校正优先回放的步长减小 4 倍。与未校正优先回放相比，重要性采样使学习不那么激进，一方面导致更慢的初始学习，另一方面降低过早收敛的风险，有时获得更好的最终结果。与均匀回放相比，完全校正的优先化平均更优。

---

## 参考文献

Andre, David, Friedman, Nir, and Parr, Ronald. Generalized prioritized sweeping. In Advances in Neural Information Processing Systems. Citeseer, 1998.

Atherton, Laura A, Dupret, David, and Mellor, Jack R. Memory trace replay: the shaping of memory consolidation by neuromodulation. Trends in neurosciences, 38(9):560–570, 2015.

Bellemare, Marc G, Naddaf, Yavar, Veness, Joel, and Bowling, Michael. The arcade learning environment: An evaluation platform for general agents. arXiv preprint arXiv:1207.4708, 2012.

Bellemare, Marc G., Ostrovski, Georg, Guez, Arthur, Thomas, Philip S., and Munos, Rémi. Increasing the action gap: New operators for reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2016.

Collobert, Ronan, Kavukcuoglu, Koray, and Farabet, Clément. Torch7: A matlab-like environment for machine learning. In BigLearn, NIPS Workshop, number EPFL-CONF-192376, 2011.

Diamond, Jared. Zebras and the Anna Karenina principle. Natural History, 103:4–4, 1994.

Felzenszwalb, Pedro, McAllester, David, and Ramanan, Deva. A discriminatively trained, multiscale, deformable part model. In CVPR 2008. IEEE, 2008.

Foster, David J and Wilson, Matthew A. Reverse replay of behavioural sequences in hippocampal place cells during the awake state. Nature, 440(7084):680–683, 2006.

Galar, Mikel, Fernandez, Alberto, Barrenechea, Edurne, Bustince, Humberto, and Herrera, Francisco. A review on ensembles for the class imbalance problem: bagging-, boosting-, and hybrid-based approaches. IEEE Transactions on SMC, Part C, 42(4):463–484, 2012.

Geramifard, Alborz, Doshi, Finale, Redding, Joshua, Roy, Nicholas, and How, Jonathan. Online discovery of feature dependencies. In ICML-11, pp. 881–888, 2011.

Guo, Xiaoxiao, Singh, Satinder, Lee, Honglak, Lewis, Richard L, and Wang, Xiaoshi. Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning. In NeurIPS 27, pp. 3338–3346, 2014.

Hinton, Geoffrey E. To recognize shapes, first learn to generate images. Progress in brain research, 165:535–547, 2007.

Kingma, Diederik P. and Ba, Jimmy. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.

Lecun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

LeCun, Yann, Cortes, Corinna, and Burges, Christopher JC. The MNIST database of handwritten digits, 1998.

Lin, Long-Ji. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3-4):293–321, 1992.

Mahmood, A Rupam, van Hasselt, Hado P, and Sutton, Richard S. Weighted importance sampling for off-policy learning with linear function approximation. In NeurIPS, pp. 3014–3022, 2014.

McNamara, Colin G, Tejero-Cantero, Álvaro, Trouche, Stéphanie, Campo-Urriza, Natalia, and Dupret, David. Dopaminergic neurons promote hippocampal reactivation and spatial memory persistence. Nature neuroscience, 2014.

Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Graves, Alex, Antonoglou, Ioannis, Wierstra, Daan, and Riedmiller, Martin. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel, Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski, Georg, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Moore, Andrew W and Atkeson, Christopher G. Prioritized sweeping: Reinforcement learning with less data and less time. Machine Learning, 13(1):103–130, 1993.

Nair, Arun, Srinivasan, Praveen, Blackwell, Sam, et al. Massively parallel methods for deep reinforcement learning. arXiv preprint arXiv:1507.04296, 2015.

Narasimhan, Karthik, Kulkarni, Tejas, and Barzilay, Regina. Language understanding for text-based games using deep reinforcement learning. In EMNLP, 2015.

Ólafsdóttir, H Freyja, Barry, Caswell, Saleem, Aman B, Hassabis, Demis, and Spiers, Hugo J. Hippocampal place cells construct reward related sequences through unexplored space. Elife, 4:e06063, 2015.

Riedmiller, Martin. Rprop-description and implementation details. 1994.

Rosin, Christopher D and Belew, Richard K. New methods for competitive coevolution. Evolutionary Computation, 5(1):1–29, 1997.

Schaul, Tom, Zhang, Sixin, and Lecun, Yann. No more pesky learning rates. In ICML-13, pp. 343–351, 2013.

Schmidhuber, Jürgen. Curious model-building control systems. In IEEE IJCNN, pp. 1458–1463, 1991.

Singer, Annabelle C and Frank, Loren M. Rewarded outcomes enhance reactivation of experience in the hippocampus. Neuron, 64(6):910–921, 2009.

Stadie, Bradly C, Levine, Sergey, and Abbeel, Pieter. Incentivizing exploration in reinforcement learning with deep predictive models. arXiv preprint arXiv:1507.00814, 2015.

Sun, Yi, Ring, Mark, Schmidhuber, Jürgen, and Gomez, Faustino J. Incremental basis construction from temporal difference error. In ICML-11, pp. 481–488, 2011.

van Hasselt, Hado. Double Q-learning. In NeurIPS, pp. 2613–2621, 2010.

van Hasselt, Hado, Guez, Arthur, and Silver, David. Deep Reinforcement Learning with Double Q-learning. In AAAI, 2016.

van Seijen, Harm and Sutton, Richard. Planning by prioritized sweeping with small backups. In ICML, pp. 361–369, 2013.

Wang, Z., de Freitas, N., and Lanctot, M. Dueling network architectures for deep reinforcement learning. Technical report, 2015.

Watkins, Christopher JCH and Dayan, Peter. Q-learning. Machine learning, 8(3-4):279–292, 1992.

White, Adam, Modayil, Joseph, and Sutton, Richard S. Surprise and curiosity for big data robotics. In Workshops at AAAI, 2014.
