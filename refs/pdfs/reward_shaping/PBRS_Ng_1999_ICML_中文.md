# 奖励变换下的策略不变性：理论及其在奖励塑形中的应用

**Policy invariance under reward transformations: Theory and application to reward shaping**

---

**Andrew Y. Ng, Daishi Harada, Stuart Russell**

计算机科学系（Computer Science Division）
加州大学伯克利分校（University of California, Berkeley）
Berkeley CA 94720
{ang, daishi, russell}@cs.berkeley.edu

---

## 摘要

本文研究了在何种条件下对马尔可夫决策过程（Markov Decision Process, MDP）的奖励函数（reward function）进行修改后，最优策略（optimal policy）能够保持不变。研究表明，除了效用理论（utility theory）中已知的正线性变换（positive linear transformation）之外，还可以为状态之间的转移添加一个奖励，只要该奖励可以表示为某个**任意**势函数（potential function）在这些状态上的值之差。进一步地，本文证明这是不变性的一个**必要**条件，即除非对底层 MDP 做出更多假设，否则任何其他变换都可能导致次优策略（suboptimal policy）。这些结果阐明了**奖励塑形**（reward shaping）这一实践的本质——奖励塑形是强化学习（reinforcement learning）中的一种方法，通过提供额外的训练奖励来引导学习智能体。特别地，本文展示了奖励塑形过程中一些众所周知的"缺陷"（bugs）源于非基于势函数的奖励，并给出了构造塑形势函数的方法，这些方法对应于基于距离（distance-based）和基于子目标（subgoal-based）的启发式方法。我们展示了这种势函数可以显著缩短学习时间。

---

## 1 引言

在序贯决策问题（sequential decision problems）中，例如动态规划（dynamic programming）和强化学习文献中所研究的问题，"任务"由**奖励函数**表示。给定奖励函数和领域模型，最优策略即被确定。一个基本的理论问题由此产生：在指定奖励函数时，我们有多大的自由度，才能使最优策略保持不变？

在效用理论领域，该理论主要研究单步决策（single-step decisions），关于效用函数的对应问题可以非常简单地回答。对于无不确定性的单步决策，效用上的任何单调变换（monotonic transformation）都不会改变最优决策；在存在不确定性的情况下，只允许正线性变换 [von Neumann and Morgenstern, 1944]。这些结果对于博弈中的评估函数设计、从人类获取效用函数以及许多其他领域都有重要意义。

据我们所知，奖励函数变换下的策略不变性问题，在序贯决策问题中尚未被充分探讨。^1 策略保持变换（policy-preserving transformations）至少在以下领域具有重要意义：

- MDP 的**结构估计**（structural estimation）任务 [Rust, 1994] 涉及从观察到的最优行为中恢复模型和奖励函数。（另见 [Russell, 1998] 中关于**逆强化学习**（inverse reinforcement learning）的讨论。）策略保持变换决定了奖励函数可被恢复的程度。

- 强化学习中**奖励塑形**的实践，是向学习智能体提供额外奖励以引导其学习过程，超越底层 MDP 所提供的奖励。理解塑形对所学策略的影响非常重要。

> ^1 已知一些关于近似不变性的结果：如果奖励被最多 $\varepsilon$ 扰动，新策略的值在原始最优策略的 $2\varepsilon/(1-\gamma)$ 范围内 [Singh and Yee, 1994, Williams and Baird, 1994]。

本文主要聚焦于奖励塑形，该方法有潜力成为将强化学习方法扩展到处理复杂问题的一项非常强大的技术 [Dorigo and Colombetti, 1994, Mataric, 1994, Randlov and Alstrom, 1998]。（类似的思想也出现在动物训练文献中，参见 [Saksida et al., 1997] 的讨论。）通常，一个非常简单的额外奖励模式就足以使一个原本完全难以处理的问题变得直截了当。

为了理解策略不变性在塑形中为何重要，考虑以下可能出现的缺陷示例：[Randlov and Alstrom, 1998] 描述了一个学习骑模拟自行车到特定位置的系统。为了加速学习，他们在智能体朝目标前进时提供正奖励。结果智能体学会了在起始状态附近骑小圈，因为靠近起始状态时没有远离目标的惩罚。智能体由此获得了正奖励，因为它碰巧在朝目标方向移动。类似的问题也出现在由 David Andre 和 Astro Teller（私人交流）训练的足球机器人上。由于控球在足球中很重要，他们为触球提供了奖励。智能体学到的策略是停留在球旁边并"振动"，尽可能频繁地触球。这些策略显然不是原始 MDP 的最优策略。

这些例子表明，塑形奖励必须遵守某些条件，否则可能误导智能体学习次优策略。正奖励循环的困难引导我们考虑从保守势函数（conservative potential）导出的奖励——即执行两个状态之间转移的奖励本质上是应用于每个状态的势函数值之差。事实证明，这不仅是保证奖励变换下策略不变性的**充分**条件，而且假设对 MDP 没有先验知识，这也是一个**必要**条件。第 2 节给出精确陈述所需的定义，第 3 节陈述并证明该结论。第 4 节展示如何构造各类塑形势函数，并在一些简单领域上演示其加速学习的效果。最后，第 5 节将我们的结果与现有算法如优势学习（Advantage learning）[Baird, 1994] 和 $\lambda$-策略迭代（$\lambda$-policy iteration）[Bertsekas and Tsitsiklis, 1996] 相联系，并以讨论和未来工作作结。

---

## 2 预备知识

### 2.1 定义

本节提供论文中使用的一些定义，聚焦于有限状态马尔可夫决策过程（MDP）的情形。塑形同时适用于有限状态和无限状态问题，但无限状态情形的底层 MDP 理论即使在没有塑形的情况下也显著更为困难。尽管如此，我们的分析和方法论在底层 MDP 理论从有限状态推广到无限状态之后，可以很容易地进行推广，我们稍后会再次提及这一点；但现在，我们先从仅显式考虑有限状态领域的定义开始。

一个（有限状态）**马尔可夫决策过程（MDP）** 是一个元组 $M = (S, A, T, \gamma, R)$，其中：$S$ 是有限**状态**（states）集合；$A = \{a_1, \ldots, a_k\}$ 是 $k \geq 2$ 个**动作**（actions）的集合；$T = \{P_{sa}(\cdot) | s \in S, a \in A\}$ 是下一状态**转移概率**（transition probabilities），$P_{sa}(s')$ 给出在状态 $s$ 采取动作 $a$ 后转移到状态 $s'$ 的概率；$\gamma \in (0, 1]$ 是**折扣因子**（discount factor）；$R$ 指定奖励分布。为简便起见，我们假设奖励是确定性的，此时 $R$ 是一个有界实值函数，称为**奖励函数**。在文献中，奖励函数通常写为 $R: S \times A \mapsto \mathbb{R}$，$R(s, a)$ 为在状态 $s$ 采取动作 $a$ 时获得的奖励。虽然我们经常使用这种形式，但也允许更一般的形式 $R: S \times A \times S \mapsto \mathbb{R}$，$R(s, a, s')$ 为在状态 $s$ 采取动作 $a$ 并转移到状态 $s'$ 时获得的奖励。

给定固定的动作集 $A$，集合 $S$ 上的**策略**（policy）是任意函数 $\pi: S \to A$。注意策略定义在状态上而非 MDP 上，因此同一策略可以应用于两个不同的 MDP，只要两个 MDP 使用相同的状态和动作。给定任意策略 $\pi$ 在状态集 $S$ 上以及任意 MDP $M = (S, A, T, \gamma, R)$（使用相同的状态和动作），我们定义**价值函数**（value function） $V_M^\pi$，在任意状态 $s$ 处求值为 $V_M^\pi(s) = \mathrm{E}[r_1 + \gamma r_2 + \gamma^2 r_3 + \ldots; \pi, s]$，其中 $r_i$ 是从状态 $s$ 开始执行策略 $\pi$ 的第 $i$ 步获得的奖励，期望取自执行 $\pi$ 时的状态转移。我们定义**最优价值函数**为 $V_M^*(s) = \sup_\pi V_M^\pi(s)$，**$Q$ 函数**在任意 $s \in S, a \in A$ 处求值为

$$Q_M^\pi(s, a) = \mathrm{E}_{s' \sim P_{sa}(\cdot)} \left[ R(s, a, s') + \gamma V_M^\pi(s') \right] \tag{1}$$

（其中记号 $s' \sim P_{sa}(\cdot)$ 表示 $s'$ 按分布 $P_{sa}(\cdot)$ 抽取），以及**最优 $Q$ 函数** $Q_M^*(s, a) = \sup_\pi Q_M^\pi(s, a)$。最后，我们定义 MDP $M$ 的**最优策略**为 $\pi_M^*(s) = \arg\max_{a \in A} Q_M^*(s, a)$。最优策略可能不唯一，更一般地，如果 $\pi(s) \in \arg\max_{a \in A} Q_M^*(s, a)$ 对所有 $s \in S$ 成立，则称策略 $\pi$ 在 $M$ 中是最优的。最后，当 MDP 上下文明确时，我们可以省略 $M$ 下标，写 $V^\pi$ 而非 $V_M^\pi$ 等。

我们还需要一些（大部分是标准的）正则条件，以确保上述所有定义有意义。对于无折扣（$\gamma = 1$）MDP，我们假设 $S$ 包含一个特殊的状态 $s_0$，称为**吸收状态**（absorbing state），使得 MDP 在转移到 $s_0$ 后"停止"，之后不再有奖励。此外，对于无折扣 MDP，我们假设所有策略都是**正则的**（proper），即从任何状态开始执行任何策略，都将以概率 1 最终转移到 $s_0$。由于这实际上是对 $T$ 的条件，我们在本文中称转移概率 $T$ 在满足此条件时为正则的。折扣 MDP 没有对应的吸收状态，且总是无限时域的（infinite-horizon）；注意因此对于折扣 MDP，我们可以写 $S - \{s_0\} = S$。

以上是有限状态 MDP 所需的标准正则条件（参见如 [Sutton and Barto, 1998]）。对于无限状态空间的 MDP，需要更多条件：例如在无折扣情形中，$V_M^\pi(s) = \mathrm{E}[r_1 + \gamma r_2 + \gamma^2 r_3 + \ldots; \pi, s]$ 中的期望可能甚至不存在。^2 这些问题需要在定义最优策略等概念之前得到妥善处理，相关的优秀参考文献包括 [Bertsekas, 1995, Hernandez-Lerma, 1989, Bertsekas and Shreve, 1978]。但不幸的是，完整解释无限 $|S|$ 情形需要比我们此处希望深入的更多测度论知识，我们仅指出，在适当推广 MDP 所需正则条件之后，所有结果都可推广到无限 $|S|$ 情形。在本文中，我们将持续指出结果如何可为无限 $|S|$ 证明，但将无限 $|S|$ 的更一般证明留给完整论文。目前，我们仅注意到对于无限状态情形，一个重要且有用的条件是：强化信号在绝对值上有界；这一点将在后文再次提及。

> ^2 这类似于柯西分布的"均值"不存在。

### 2.2 塑形奖励

本节介绍我们关于塑形奖励（shaping rewards）的正式框架。直觉上，我们试图为某个 MDP $M = (S, A, T, \gamma, R)$ 学习一个策略，并希望通过给学习算法提供额外的"塑形"奖励来帮助它，从而将其引导至更快地学到一个好的（或最优的）策略。为形式化这一点，我们假设不是在 $M = (S, A, T, \gamma, R)$ 上运行强化学习算法，而是在某个**变换后的** MDP $M' = (S, A, T, \gamma, R')$ 上运行，其中 $R' = R + F$ 是变换后 MDP 的奖励函数，$F: S \times A \times S \mapsto \mathbb{R}$ 是一个有界实值函数，称为**塑形奖励函数**（shaping reward function）。（类似于 $R$，对于无折扣情形 $F$ 的定义域严格来说应为 $S - \{s_0\} \times A \times S$，但我们不在此过于拘泥。）因此，如果在原始 MDP $M$ 中从 $s$ 采取动作 $a$ 转移到 $s'$ 会获得奖励 $R(s, a, s')$，那么在新 MDP $M'$ 中同一事件会获得奖励 $R(s, a, s') + F(s, a, s')$。

对于任何固定 MDP，假设塑形奖励是加性的且无记忆的（memoryless），则 $R' = R + F$ 是塑形奖励的最一般形式。^3 此外，它们涵盖了人们可能想到的相当广泛的塑形奖励。例如，为了鼓励朝目标移动，可以选择 $F(s, a, s') = r$（当 $s'$ 在某种适当意义下比 $s$ 更接近目标时），否则 $F(s, a, s') = 0$，其中 $r$ 是某个正奖励。或者，为了鼓励在某个状态集 $S_0$ 中采取动作 $a_1$，可以设 $F(s, a, s') = r$（当 $a = a_1, s \in S_0$ 时），否则 $F(s, a, s') = 0$。

> ^3 在完整论文中，我们将考虑一种更一般的、不一定是加性的形式：$R'(s, a, s') = F(r, s, a, s')$，其中 $F$ 是任意函数，$r = R(s, a, s')$ 是在原始 MDP $M$ 中将获得的奖励。在适当条件下，如果我们要给出与此处类似的最优性保证，那么选择塑形奖励的唯一额外自由度是允许我们将奖励乘以任意固定正因子。由于这不增加任何有趣的丰富性，我们将此结果留给完整论文。

这种奖励变换形式的一个基本但重要的性质是它可以**被实现**：在许多强化学习应用中，我们并不被显式给定 $M$ 的元组 $(S, A, T, \gamma, R)$，而是只允许通过在 MDP 中采取动作并观察产生的状态转移和奖励来学习 $M$。给定对 $M$ 的这种访问方式，我们可以模拟对 $M'$ 的同类型访问，方法是在 $M$ 中采取动作，然后在实际观察到奖励 $R(s, a, s')$ 时"假装"观察到了奖励 $R(s, a, s') + F(s, a, s')$。这能够成立的原因很自然：$M$ 和 $M'$ 使用相同的动作、状态和转移概率。因此，可以应用于 $M$ 的在线/离线的基于模型/无模型算法通常也可以同样方式应用于 $M'$。

由于我们是在 $M'$ 中学习策略以期望在 $M$ 中使用，因此面临的问题是：对于什么形式的塑形奖励函数 $F$，我们能保证 $\pi_{M'}^*$（$M'$ 中的最优策略）在 $M$ 中也是最优的？下一节将以相当高的一般性回答这个问题。

---

## 3 主要结果

在实际应用中，我们通常不精确知道 $T$（也可能不知道 $R(s, a, s')$）。因此我们的目标是，给定 $S$ 和 $A$（可能还有 $R$），找到一个塑形奖励函数 $F: S \times A \times S \mapsto \mathbb{R}$，使得它足够"好"，从而 $\pi_{M'}^*$ 在 $M$ 中也是最优的。本节给出一种 $F$ 的形式，在此形式下可以保证 $\pi_{M'}^*$ 在 $M$ 中最优。我们还提供了一个弱逆命题，表明在不进一步了解 $T$ 和 $R$ 的情况下，这是唯一能始终给出此保证的塑形函数类型。

首先聚焦于无折扣情形（$\gamma = 1$），让我们尝试获得一些关于什么样的 $F$ 可能导致引言中提到的塑形"缺陷"的直觉。在 Randlov 和 Alstrom 的自行车任务中，智能体因朝目标骑行而获得奖励但不因远离目标而受惩罚，它学会了骑小圈，并在恰好朝目标方向移动时获得正奖励。更一般地，如果存在某个状态序列 $s_1, s_2, \ldots, s_n$ 使得智能体可以循环通过（$s_1 \to s_2 \to \cdots \to s_n \to s_1 \to \cdots$），并通过如此获得正的净塑形奖励（$F(s_1, a_1, s_2) + \cdots + F(s_{n-1}, a_{n-1}, s_n) + F(s_n, a_n, s_1) > 0$），那么智能体可能会从它真正应该做的事情（如朝目标骑行）中被"分心"，转而反复沿着这个循环运动。

为解决循环问题，一个立即想到的 $F$ 形式是让 $F$ 为**势之差**（difference of potentials）：$F(s, a, s') = \Phi(s') - \Phi(s)$，其中 $\Phi$ 是状态上的某个函数。这样，$F(s_1, a_1, s_2) + \cdots + F(s_{n-1}, a_{n-1}, s_n) + F(s_n, a_n, s_1) = 0$，我们就消除了"分心"智能体的循环问题。还有其他选择 $F$ 的方式吗？除了避免循环之外，塑形还需要解决其他问题吗？事实证明，在对 $T$ 和 $R$ 没有更多先验知识的情况下，这种基于势的塑形函数 $F$ 是唯一能保证与 $M$ 中最优策略一致的 $F$。此外，这本质上就是做出此保证所需的全部。这在以下定理中正式表述：

**定理 1** *设任意 $S$、$A$、$\gamma$ 和任意塑形奖励函数 $F: S \times A \times S \mapsto \mathbb{R}$ 给定。我们称 $F$ 是**基于势的**（potential-based）塑形函数，如果存在一个实值函数 $\Phi: S \mapsto \mathbb{R}$ 使得对所有 $s \in S - \{s_0\}, a \in A, s' \in S$，*

$$F(s, a, s') = \gamma\Phi(s') - \Phi(s), \tag{2}$$

*（其中当 $\gamma < 1$ 时 $S - \{s_0\} = S$）。那么，$F$ 是基于势的塑形函数是保证其与最优策略一致的**充分且必要**条件（当从 $M' = (S, A, T, \gamma, R+F)$ 而非 $M = (S, A, T, \gamma, R)$ 学习时），含义如下：*

- *（充分性）如果 $F$ 是基于势的塑形函数，则 $M'$ 中的每个最优策略在 $M$ 中也是最优的（反之亦然）。*

- *（必要性）如果 $F$ 不是基于势的塑形函数（即不存在满足公式 (2) 的 $\Phi$），则存在（正则的）转移函数 $T$ 和奖励函数 $R: S \times A \mapsto \mathbb{R}$，使得 $M'$ 中没有最优策略在 $M$ 中是最优的。*

另外注意以下几点：对于无限状态情形，如果要选择某个 $\Phi$ 来构造基于势的塑形函数并使形式化结果成立，我们确实需要要求 $\Phi$ 有界，从而塑形奖励 $F$ 也有界（类似于第 2.1 节中 $R$ 有界的条件）；这个问题将在后文再次讨论。注意对于有限状态情形，这是一个空条件，因为 $\Phi$ 的值域具有有限基数，自动有界。此外，上述充分和必要条件可能看起来比通常复杂一些，这是因为 $M$ 或 $M'$ 中可能存在多个最优策略。尽管如此，这些量化条件使该定理成为此类定理中最强的形式。充分性条件表明，只要使用基于势的 $F$，我们就保证任何 $\pi_{M'}^*$（我们可能试图学习的）在 $M$ 中也是最优的。必要性条件表明，如果我们对 $T$ 和 $R$ 没有任何知识，那么如果我们希望保证与 $M$ 中最优策略的一致性，就必须为 $M'$ 中的学习选择基于势的 $F$。（如果我们确实对 $T$ 和 $R$ 有深入了解，那么必要性条件就不那么重要了，我们可能可以使用其他塑形函数。）

必要性的证明在附录 A 中给出。这里我们只证明公式 (2) 是充分条件：即如果 $F$ 确实具有 (2) 中的形式，则可以保证 $M'$ 中的每个最优策略在 $M$ 中也是最优的。同样，我们仅对有限 $|S|$ 情形进行严格证明；无限 $|S|$ 的证明几乎相同，只是在使用贝尔曼方程（Bellman Equations）的论证中需要稍多注意。

**充分性证明：** 设 $F$ 具有 (2) 的形式。若 $\gamma = 1$，由于将 $\Phi(s)$ 替换为 $\Phi'(s) = \Phi(s) - k$（对任意常数 $k$）不会改变塑形奖励 $F$（它是势之差），我们可以在必要时将 $\Phi(s)$ 替换为 $\Phi(s) - \Phi(s_0)$，从而不失一般性地假设用于通过 (2) 表示 $F$ 的 $\Phi$ 满足 $\Phi(s_0) = 0$。

对于原始 MDP $M$，我们知道其最优 $Q$ 函数 $Q_M^*$ 满足贝尔曼方程（参见如 [Sutton and Barto, 1998]）：

$$Q_M^*(s, a) = \mathrm{E}_{s' \sim P_{sa}(\cdot)} \left[ R(s, a, s') + \gamma \max_{a' \in A} Q_M^*(s', a') \right]$$

一些简单的代数运算给出：

$$Q_M^*(s, a) - \Phi(s) = \mathrm{E}_{s'} \left[ R(s, a, s') + \gamma\Phi(s') - \Phi(s) + \gamma \max_{a' \in A} (Q_M^*(s', a') - \Phi(s')) \right]$$

如果我们现在定义 $\tilde{Q}_{M'}(s, a) \triangleq Q_M^*(s, a) - \Phi(s)$ 并将 $F(s, a, s') = \gamma\Phi(s') - \Phi(s)$ 代回上式，得到：

$$\tilde{Q}_{M'}(s, a) = \mathrm{E}_{s'} \left[ R(s, a, s') + F(s, a, s') + \gamma \max_{a' \in A} \tilde{Q}_{M'}(s', a') \right] = \mathrm{E}_{s'} \left[ R'(s, a, s') + \gamma \max_{a' \in A} \tilde{Q}_{M'}(s', a') \right]$$

但这恰好是 $M'$ 的贝尔曼方程。对于无折扣情形，我们还有 $\tilde{Q}_{M'}(s_0, a) = Q_M^*(s_0, a) - \Phi(s_0) = 0 - 0 = 0$。所以 $\tilde{Q}_{M'}(s, a)$ 满足 $M'$ 的贝尔曼方程，因此必然是 $M'$ 的唯一最优 $Q$ 函数。于是 $Q_{M'}^*(s, a) = Q_M^*(s, a) - \Phi(s)$，$M'$ 的最优策略因此满足：

$$\pi_{M'}^*(s) \in \arg\max_{a \in A} Q_{M'}^*(s, a) = \arg\max_{a \in A} Q_M^*(s, a) - \Phi(s) = \arg\max_{a \in A} Q_M^*(s, a)$$

因此在 $M$ 中也是最优的。为证明 $M$ 中的每个最优策略在 $M'$ 中也是最优的，只需将上述证明中 $M$ 和 $M'$ 的角色互换（并使用塑形函数 $-F$）即可。证毕。 $\square$

**推论 2** *在定理 1 的条件下，假设 $F$ 确实具有形式 $F(s, a, s') = \gamma\Phi(s') - \Phi(s)$。进一步假设当 $\gamma = 1$ 时 $\Phi(s_0) = 0$。则对所有 $s \in S$，$a \in A$，*

$$Q_{M'}^*(s, a) = Q_M^*(s, a) - \Phi(s), \tag{3}$$

$$V_{M'}^*(s) = V_M^*(s) - \Phi(s). \tag{4}$$

**证明：** (3) 已在上述充分性证明中证明；(4) 由恒等式 $V^*(s) = \max_{a \in A} Q^*(s, a)$ 直接得出。 $\square$

**注释 1（鲁棒性与学习）：** 虽然我们在此未证明，但推论 2 中的恒等式实际上对任意策略 $\pi$ 成立，而不仅仅是最优策略：$V_{M'}^\pi(s) = V_M^\pi(s) - \Phi(s)$（$Q$ 函数类似）。由此可得，基于势的塑形是**鲁棒的**（robust），即近最优策略也被保持；也就是说，如果我们在 $M'$ 中使用基于势的塑形学到一个近最优策略 $\pi$（即 $|V_{M'}^*(s) - V_{M'}^\pi(s)| < \varepsilon$），则 $\pi$ 在 $M$ 中也是近最优的（$|V_M^*(s) - V_M^\pi(s)| < \varepsilon$）。（为此，应用上述恒等式于策略 $\pi$ 和 $\pi_M^* = \pi_{M'}^*$，然后相减。）

**注释 2（$\Phi$ 下所有策略最优）：** 为更好地理解为何基于势的 $F$ 保持最优策略，值得注意的是：如果一个 MDP $M$ 具有基于势的**强化函数**（reinforcement function） $R(s, a, s') = \gamma\Phi(s') - \Phi(s)$，则**任何**策略在 $M$ 中都是最优的。因此，基于势的塑形函数对策略是无差别的，即它们不给我们任何理由偏好某个策略而非其他；在直觉层面，这解释了为何当我们从 $M$ 切换到 $M'$ 时，它们不给我们任何理由偏好 $\pi_M^*$ 以外的策略。

该定理建议我们选择形如 $F(s, a, s') = \gamma\Phi(s') - \Phi(s)$ 的塑形奖励。在应用中，$\Phi$ 当然应利用关于领域的专家知识来选择。关于如何选择，推论 2 给出了一个特别好的 $\Phi$ 形式：如果我们对领域足够了解，可以尝试选择 $\Phi(s) = V_M^*(s)$（在无折扣情形中取 $\Phi(s_0) = 0$），则公式 (4) 告诉我们 $M'$ 中的价值函数为 $V_{M'}^*(s) \equiv 0$——这是一个特别容易学习的价值函数；即使缺乏世界模型，所需做的全部就是学习非零的 $Q$ 值。为避免误解，我们也强调这不是选择有用 $\Phi$ 的唯一方式，即使 $\Phi$ **远离** $V_M^*$（比如在上确范数意义下），塑形奖励也能显著帮助，例如通过引导探索等，我们将在下一节看到相关示例。但无论如何，只要我们选择基于势的 $F$，就能保证在 $M'$ 中学到的任何（近）最优策略在 $M$ 中也是（近）最优的。现在让我们将注意力转向一些展示基于势的塑形如何在实践中应用的小实验。

---

## 4 实验

在我们之前已有大量实证工作令人信服地证明了塑形的有效性 [Mataric, 1994, Randlov and Alstrom, 1998]，我们不打算进一步证明其有效性。相反，我们的目标是展示基于势的塑形函数如何融入整体框架，并演示如何推导出此类塑形函数。

为此，为了简洁和清晰，我们选择使用非常简单的网格世界（grid-world）领域来展示基于势的塑形的有趣特性。第一个领域是一个最短路径到目标的 10x10 网格世界，起始状态和目标状态位于对角，无折扣，每步强化为 -1。动作为 4 个罗盘方向，80% 的概率朝预期方向移动 1 步，20% 的概率朝随机方向移动，如果试图走出网格则留在原地。什么是好的塑形势函数 $\Phi(s)$？我们前面指出公式 (4) 建议 $\Phi(s) = V_M^*(s)$ 可能是好的塑形势函数。因此让我们通过可能得出 $V_M^*$ 的粗略估计的推理来演示，希望展示如何利用关于距离和目标位置的少量专家知识，通过类似推理为其他最小代价到目标问题推导出 $\Phi$。

在尝试朝目标迈步时，有 80% 的机会成功，20% 的机会做出随机动作。如果做出随机动作，除非在网格边界，否则朝目标和远离目标的可能性相同。因此，从大多数状态出发，最优策略预期每个时步朝目标前进约 0.8 步（曼哈顿距离，Manhattan distance）。从状态 $s$ 到达目标的期望步数的粗略估计为 $\text{MANHATTAN}(s, \text{GOAL}) / 0.8$。因此，我们将价值函数的估计设为 $\Phi_0(s) = \tilde{V}_M(s) = -\text{MANHATTAN}(s, \text{GOAL}) / 0.8$。这就是我们用作"好"塑形函数的猜测。此外，作为一个在上确范数意义下远离 $V_M^*(s)$ 的塑形奖励，我们也尝试了 $\Phi(s) = 0.5\Phi_0(s)$。

> **图 1 描述：** (a) 10x10 网格世界实验。到达目标所需步数与试验次数的关系图。虚线为无塑形，点划线为 $\Phi = 0.5\Phi_0$，实线为 $\Phi = \Phi_0$。(b) 50x50 网格世界实验。
>
> 所有实验结果为 40 次独立运行的平均值。^4

如图 1a 所示，使用这两种塑形函数都显著加速了学习。而且值得再次强调的是，即使 $0.5\Phi_0$ 与 $V_M^*$ 相差甚远，它仍然显著帮助了学习的初始阶段。对于更大的 50x50 网格世界，结果更加显著：图 1b 展示了相同实验在更大网格上重复的结果。$\Phi_0$ 和 $0.5\Phi_0$ 的曲线在图中太低以至于几乎看不到；无塑形的学习显然远远落后于基于势的塑形算法。

> ^4 使用 Sarsa [Sutton and Barto, 1998]，0.10-贪心探索，学习率 0.02。使用 Sarsa($\lambda$) 的实验也给出了类似结果，塑形显著加速了学习。

重申一下，这些实验的目标不是试图证明塑形的有效性——这已被其他人更有说服力地证明。相反，我们在这里展示的是一种非常简单的推理风格，通过组合一个到目标距离的启发式方法，使我们能够选择一个显著加速学习的合理 $\Phi$。

接下来，另一类适用类似推理的问题是可以分配子目标（subgoals）的领域。考虑图 2a 中的网格世界，我们从左下角出发，必须按顺序 1, 2, 3, 4, $G$ 拾取一组"旗帜"，然后到达最终目标状态。动作和奖励与前一个网格世界相同，状态空间扩展以跟踪已收集旗帜的情况。由于每个旗帜是一个子目标，很自然地选择 $F$ 使我们因访问子目标而获得奖励。让我们看看势函数的推理如何引导我们选择这样的 $F$，以及公式 (4) 如何进一步建议子目标奖励的大小。

利用子目标位置的知识以及前面建议的类似推理（每时步 0.8 步进展等），我们可以估计到达目标所需的期望时步数 $t$。如果我们想象每个子目标大约同样难以从前一个到达，那么到达第 $n$ 个子目标后，还需约 $((5-n)/5)t$ 步。稍微细化的论证将其改为 $((5-n-0.5)/5)t$ 步（其中 0.5 来自我们处于第 $n$ 个和第 $n+1$ 个子目标之间的"典型情况"），因此我们的第一选择 $\Phi_0(s) = -((5 - n_s - 0.5)/5)t$，其中 $n_s$ 表示状态 $s$ 处已实现的子目标数量。使用这种塑形奖励函数形式，我们看到 $\Phi(s) = \Phi_0(s)$ 在每次到达任何子目标时（最终目标状态除外）跳跃 $t/5$，因此塑形奖励函数 $F(s, a, s') = \Phi(s') - \Phi(s)$ 为到达每个子目标给予 $t/5$ 的奖励。这正是我们的直觉认为可能是好的塑形奖励。

> **图 2 描述：** (a) 5x5 网格世界，有 5 个子目标（包括目标状态），必须按 1, 2, 3, 4, $G$ 顺序访问。(b) 带子目标的 5x5 网格世界实验。到达目标所需步数与试验次数的关系图。虚线为无塑形，点划线为 $\Phi = \Phi_0$，实线为 $\Phi = \Phi_1$。

作为对比，我们还使用一个更精细调整的塑形奖励进行了实验，类似于前面的网格世界实验，显式估计每个状态到目标的剩余时间，并构造相应的 $\Phi_1(s) = \tilde{V}_M(s)$ 势函数。这些实验的结果如图 2b 所示，我们看到即使使用我们的第一个粗糙塑形函数 $\Phi_0$，也已经显著加速了学习（精细调整的 $\Phi_1$ 不出意料地给出了更好的表现）。在更大的领域或更多子目标的情况下重复此实验，结果更加显著。

---

## 5 讨论与结论

我们已经证明了塑形函数 $F$ 使最优策略保持不变的充分且必要条件。这里值得提及两个简单的推广：除了在试图学习最优策略时保证一致性之外，容易证明（通过类似于第 3 节注释 1 的论证）基于势的 $F$ 在从**受限**（restricted）策略类中学习好策略时也有效，例如 [Kearns et al., 1999] 研究的框架（其中包括为从状态到动作的神经网络找到最佳权重的任务）。此外，对于半马尔可夫决策过程（Semi-Markov Decision Processes, SMDPs），其中动作需要不同时间完成，公式 (2) 不出意料地推广为 $F(s, a, s', \tau) = e^{-\beta\tau}\Phi(s') - \Phi(s)$，其中 $\tau$ 是动作完成所需时间，$\beta$ 是折扣率。

最后，"$\gamma\Phi(s') - \Phi(s)$" 的形式表面上与优势学习（Advantage learning）[Baird, 1994] 和 $\lambda$-策略迭代 [Bertsekas and Tsitsiklis, 1996] 中使用的一些方程相似。在非常粗糙的层面上，这些方法中的每一种都可以被看作是尝试修改 $\Phi$ 以获得某种计算或表示上的优势。如果我们考虑修改 $\Phi$ 的问题，那么尝试学习一个粗糙的塑形函数似乎自然地导向一种多尺度值函数逼近算法；虽然尝试*学习*塑形函数最初可能显得不寻常，但正是其中多尺度"粗到细"的逼近特性使其可能非常强大；^5 这将是未来工作的主题。

> ^5 这也与以下观察有关：学习到的塑形奖励似乎在心理层面上起作用——例如国际象棋中吃子被视为一种奖励，尽管底层 MDP 仅在将杀时有奖励。

在本文中，我们已证明基于势的塑形奖励 $\gamma\Phi(s') - \Phi(s)$ 保持（近）最优策略不变。此外，这被证明是唯一能保证此不变性的塑形类型，除非我们对 MDP 做进一步假设。但正如一些实践者即使在无折扣问题上也使用折扣（也许为了改善算法的收敛性），我们相信未来使用基于势的塑形奖励的经验也可能导致人们偶尔尝试受势启发但不严格符合我们给出形式的塑形奖励。例如，类比于即使在无折扣问题上也使用折扣，可以想见对于某些问题，专家可能更容易提出一个"无折扣"的势 $\Phi$ 作为塑形函数 $\Phi(s') - \Phi(s)$，即使 $\gamma \neq 1$。虽然我们的定理此时可能不再保证最优性，但从工程角度来看，这种塑形函数可能仍然值得审慎地尝试。同样地，虽然我们的正则条件要求有界的 $\Phi$，但一些实践者可能想尝试某些无界的 $\Phi$。自然地，如果关于领域的专家知识可用，那么非基于势的塑形函数也可能完全合适。

作为选择塑形函数的指导方针，我们提出了一种基于距离的启发式方法和一种基于子目标的启发式方法来选择势函数；由于塑形对于使学习变得可行往往至关重要，我们相信寻找好的塑形函数将是一个日益重要的问题。

---

## 致谢

A. Ng 获 Berkeley Fellowship 资助。本工作还得到 ARO MURI grant DAAH04-96-1-0341、ONR grant N00014-97-1-0941 和 NSF grant ECS-9873474 的部分支持。

---

## 参考文献

[Baird, 1994] Baird, L. C. (1994). Reinforcement Learning in continuous time: Advantage updating. In *Proceedings of the International Conference on Neural Networks*.

[Bertsekas, 1995] Bertsekas, D. P. (1995). *Dynamic Programming and Optimal Control, Volume II*. Athena Scientific.

[Bertsekas and Shreve, 1978] Bertsekas, D. P. and Shreve, S. E. (1978). *Stochastic Optimal Control: The Discrete Time Case*. Academic Press.

[Bertsekas and Tsitsiklis, 1996] Bertsekas, D. P. and Tsitsiklis, J. N. (1996). *Neuro-dynamic Programming*. Athena Scientific.

[Dorigo and Colombetti, 1994] Dorigo, M. and Colombetti, M. (1994). Robot shaping: Developing autonomous agents through learning. *Artificial Intelligence*, 71(2):321–370.

[Hernandez-Lerma, 1989] Hernandez-Lerma, O. (1989). *Adaptive Markov Control Processes*. Springer-Verlag.

[Kearns et al., 1999] Kearns, M., Mansour, Y., and Ng, A. Y. (1999). Approximate planning in large POMDPs via reusable trajectories. *(Preprint)*.

[Mataric, 1994] Mataric, M. J. (1994). Reward functions for accelerated learning. In *Proceedings of the Eleventh International Conference on Machine Learning*. Morgan Kaufmann.

[Randlov and Alstrom, 1998] Randlov, J. and Alstrom, P. (1998). Learning to drive a bicycle using reinforcement learning and shaping. In *Proceedings of the Fifteenth International Conference on Machine Learning*. Morgan Kaufmann.

[Russell, 1998] Russell, S. (1998). Learning agents for uncertain environments (extended abstract). In *Proceedings of the Eleventh Annual ACM Workshop on Computational Learning Theory (COLT-98)*, Madison, Wisconsin. ACM Press.

[Rust, 1994] Rust, J. (1994). Do people behave according to Bellman's principal of optimality? Submitted to Journal of Economic Perspectives.

[Saksida et al., 1997] Saksida, L., Raymond, S., and Touretzky, D. (1997). Shaping robot behaviour using principles from instrumental conditioning. *Robotics and Autonomous Systems*, 22(3–4):231–249.

[Singh and Yee, 1994] Singh, S. and Yee, R. (1994). An upper bound on the loss from approximate optimal-value functions. *Machine Learning*, 16:227–233.

[Sutton and Barto, 1998] Sutton, R. S. and Barto, A. G. (1998). *Reinforcement Learning: An Introduction*. MIT Press.

[von Neumann and Morgenstern, 1944] von Neumann, J. and Morgenstern, O. (1944). *Theory of Games and Economic Behavior*. Princeton University Press, Princeton, New Jersey, first edition.

[Williams and Baird, 1994] Williams, R. J. and Baird, L. C. (1994). Tight performance bounds on greedy policies based on imperfect value functions. In *Proceedings of the Tenth Yale Workshop on Adaptive and Learning Systems*.

---

## 附录 A：必要性证明

本附录概述定理 1 的必要性部分的证明。为简洁起见，我们仅给出 $|A| = 2$ 情形的证明；推广是显然但更为繁琐的。我们从以下引理开始。

**引理 3** *如果存在 $s \in S - \{s_0\}$，$s' \in S$ 和 $a, a' \in A$ 使得 $F(s, a, s') \neq F(s, a', s')$，则存在（正则的）转移函数 $T$ 和奖励函数 $R$ 使得 $M'$ 中没有最优策略在 $M$ 中是最优的。*

**证明**（引理 3 概要）：不失一般性假设 $F(s, a, s') > F(s, a', s')$，令 $\Delta = F(s, a, s') - F(s, a', s') > 0$。在无折扣情形中（$\gamma = 1$），同时假设 $s \neq s'$。（当 $s = s'$ 时证明几乎相同，但为确保正则性需要更多处理。）我们构造 $M$ 如下：令 $P_{sa}(s') = P_{sa'}(s') = 1.0$，$R(s, a, s') = 0$，$R(s, a', s') = \Delta/2$。显然 $\pi_M^*(s) = a'$。另一方面，由于 $R' = R + F$，我们有 $R'(s, a, s') = F(s, a, s')$ 和 $R'(s, a', s') = \Delta/2 + F(s, a', s') = F(s, a, s') - \Delta/2 < R'(s, a, s')$，因此 $\pi_{M'}^*(s) = a$。 $\square$

我们现在准备证明主要的必要性结论。

**必要性证明。** 假设 $F$ 不是基于势的。我们需要证明可以构造 $T, R$ 使得 $M'$ 中没有最优策略 $\pi_{M'}^*$ 在 $M$ 中也是最优的。根据引理 3，如果 $F(s, a, s')$ 依赖于 $a$，则证明完成；因此我们只需考虑形如 $F(s, a, s') = F(s, s')$（不依赖于 $a$）的塑形函数。

若 $\gamma = 1$，令 $\hat{s}_0 = s_0$ 为特殊的吸收状态；否则令 $\hat{s}_0$ 为某个固定状态。注意到当 $\gamma < 1$ 时奖励的常数偏移不影响最优策略，必要时可将所有 $F(s, s')$ 替换为 $F(s, s') - F(\hat{s}_0, \hat{s}_0)$，从而不失一般性假设 $F(\hat{s}_0, \hat{s}_0) = 0$。现在定义 $\Phi(s) = -F(s, \hat{s}_0)$ 对所有 $s$。由 $F$ 不是基于势的假设，存在 $s_1, s_2$ 使得 $\gamma\Phi(s_2) - \Phi(s_1) \neq F(s_1, s_2)$（令 $s_1, s_2, \hat{s}_0$ 互不相同；其他情形类似处理）。我们按如下方式构造 $M$（仍假设 $|A| = 2$）。从状态 $s_1$ 出发，令 $P_{s_1 a}(\hat{s}_0) = P_{s_1 a'}(s_2) = 1.0$，从状态 $s_2$ 和 $\hat{s}_0$ 出发，两个动作 $a$ 和 $a'$ 都以概率 1 转移到 $\hat{s}_0$。同时定义 $\Delta = F(s_1, s_2) + \gamma F(s_2, \hat{s}_0) - F(s_1, \hat{s}_0)$，令 $R(s_1, a, \hat{s}_0) = \Delta/2$，其余 $R(\cdot, \cdot, \cdot) = 0$。

> **图 3 描述：** 未标记的粗边对应两个动作。所有边的概率均为 1。边 $(s_1, a, \hat{s}_0)$ 携带奖励 $\Delta/2$，其余边奖励为零。

则我们有

$$Q_M^*(s_1, a) = \frac{\Delta}{2}$$

$$Q_M^*(s_1, a') = 0$$

$$Q_{M'}^*(s_1, a) = \frac{\Delta}{2} + F(s_1, \hat{s}_0)$$

$$= F(s_1, s_2) + \gamma F(s_2, \hat{s}_0) - \frac{\Delta}{2}$$

$$Q_{M'}^*(s_1, a') = F(s_1, s_2) + \gamma F(s_2, \hat{s}_0),$$

其中我们利用了 $V_M^*(\hat{s}_0) = V_{M'}^*(\hat{s}_0) = 0$ 这一构造事实。因此

$$\pi_M^*(s_1) = \begin{cases} a & \text{if } \Delta > 0, \\ a' & \text{otherwise} \end{cases}$$

$$\pi_{M'}^*(s_1) = \begin{cases} a' & \text{if } \Delta > 0, \\ a & \text{otherwise} \end{cases}$$

$\square$
