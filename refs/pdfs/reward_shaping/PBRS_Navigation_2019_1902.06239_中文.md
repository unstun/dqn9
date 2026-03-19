# 一种新的基于势函数的强化学习智能体奖励塑形方法

**Babak Badnava, Nasser Mozayani**

计算机工程学院，伊朗科技大学

2017 年 11 月

arXiv:1902.06239v1 [cs.AI] 2019 年 2 月 17 日

---

## 摘要

基于势函数的奖励塑形（Potential-Based Reward Shaping, PBRS）是一类机器学习方法，旨在通过在执行任务过程中提取和利用额外知识来提高强化学习（Reinforcement Learning, RL）智能体的学习速度。迁移学习（Transfer Learning）过程包含两个步骤：从先前学习的任务中提取知识，以及将该知识迁移到目标任务中使用。后一步骤在文献中已被广泛讨论，并提出了多种方法；而前者的探索相对较少。考虑到这一点，所传递知识的类型非常重要，可能带来显著的改进。在迁移学习和基于势函数的奖励塑形的文献中，从学习过程本身收集的知识从未被涉及。本文提出了一种新颖的基于势函数的奖励塑形方法，尝试从学习过程中提取知识。所提方法从各回合的累积奖励（Episode Cumulative Reward）中提取知识。该方法在街机学习环境（Arcade Learning Environment）中进行了评估，结果表明在单任务和多任务强化学习智能体中均改善了学习过程。

**关键词**：基于势函数的奖励塑形、强化学习、奖励塑形、知识提取

---

## 1 引言

在强化学习问题中，智能体学习最大化一个可能具有时间延迟的奖励信号 [16]。近年来，RL 框架在处理复杂问题方面取得了巨大成功。然而，掌握困难任务往往很慢。因此，大多数 RL 研究者专注于通过使用由专家或启发式函数（Heuristic Function）提供的知识来提高学习过程的速度。

迁移学习是试图通过使用从先前学习任务中提取的知识来提高学习速度的方法之一 [6]。奖励塑形（Reward Shaping, RS）也是用于迁移此类知识或任何其他类型知识的方法之一。

在可以用 RL 建模的问题中，许多任务具有稀疏奖励信号（Sparse Reward Signal）。例如，有些任务中智能体在到达目标之前不会收到任何奖励，或者在环境中某些事件发生之前不会获得任何信号。

任务在环境中的奖励函数与时间无关，智能体试图在一个回合中最大化其折扣奖励之和；而从一个学习回合到另一个回合，智能体可以使用从过去回合中提取的知识来强化奖励信号。有许多信息可以用来强化奖励信号。例如，一个学习回合结束后，智能体可以将其表现与目前为止最好或最差的回合进行比较，通过这种比较来改善自身的学习过程。

---

## 2 背景

### 2.1 强化学习

RL 是一组机器学习方法，可用于利用环境反馈训练智能体。环境中的 RL 智能体可以建模为马尔可夫决策过程（Markov Decision Process, MDP）[16]。MDP 是一个元组 $(S, A, T, \gamma, R)$，其中：$S$ 是环境中出现的状态集合；$A$ 是智能体可执行动作的集合；$T: S \times A \times S \to [0,1]$ 是一个函数，给出当智能体处于状态 $s$ 并选择执行动作 $a$ 时到达状态 $s'$ 的概率；$R: S \times A \times S \to \mathbb{R}$ 是一个函数，给出智能体在状态 $s$ 执行动作 $a$ 并转移到状态 $s'$ 时获得的奖励；$\gamma$ 称为折扣因子（Discount Factor），表示未来奖励对智能体的重要程度。

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \tag{1}$$

智能体希望在其生命周期内最大化公式 (1)，即期望折扣回报（Expected Discounted Return）[16]。训练 RL 智能体有许多方法，如 Q 学习（Q-learning）[19] 和 SARSA，它们估计每个（状态，动作）对的值，并利用该估计值计算能够最大化智能体折扣回报的策略。还有一些方法使用策略梯度（Policy Gradient）[2, 17, 21] 来近似策略，并将该策略用于智能体的决策。

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right] \tag{2}$$

### 2.2 奖励塑形

奖励塑形指的是允许智能体在人工奖励信号而非环境反馈上进行训练 [18]。然而，该人工信号必须是势函数（Potential Function），否则新 MDP 的最优策略将不同 [14]。文献 [14] 的作者证明，$F$ 是势函数当且仅当存在一个实值函数 $\varphi: S \to \mathbb{R}$，使得对所有 $s \in S - \{s_0\}$，$a \in A$，$s' \in S$：

$$F(s, s') = \gamma \varphi(s') - \varphi(s) \tag{3}$$

因此，通过使用该势函数，新 MDP $M' = (S, A, T, \gamma, R+F)$ 中的最优策略对于原 MDP $M = (S, A, T, \gamma, R)$ 仍然是最优的。在新 MDP 中，公式 (3) 以及由公式 (3) 扩展的所有公式中，$\gamma$ 保持不变，必须与原 MDP 中的值相同。文献中还有其他扩展势函数的工作。文献 [20] 的作者表明势函数可以是联合状态-动作空间的函数。随后，文献 [7] 的作者表明势函数可以是动态函数，可能随时间变化，同时基于势函数的奖励塑形的所有性质保持不变。除 [20] 和 [7] 之外，文献 [12] 结合了这两种扩展，表明任何形如公式 (6) 的函数都可以是势函数。

$$F(s, a, s', a') = \gamma \varphi(s', a') - \varphi(s, a) \tag{4}$$

$$F(s, t, s', t') = \gamma \varphi(s', t') - \varphi(s, t) \tag{5}$$

$$F(s, a, t, s', a', t') = \gamma \varphi(s', a', t') - \varphi(s, a, t) \tag{6}$$

通过使用奖励塑形，Q 学习的更新规则变为公式 (7)。在公式 (7) 中，$F$ 可以是公式 (3)、(4)、(5) 或 (6) 中的任意一种。公式 (7) 为我们提供了通过向智能体提供关于问题的额外信息来增强学习的机会。这些额外知识可以来自任何来源，如人类对问题的知识、某种启发式函数，或智能体自身提取的知识。

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + F + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right] \tag{7}$$

---

## 3 相关工作

首先应考虑的是带奖励塑形的多网格 RL（Multi-grid RL with RS）方法 [11]，该方法提出以在线方式学习势函数。文献 [11] 的作者在学习过程中估计一个价值函数，并利用该价值函数构造势函数。所估计的价值函数是抽象状态（Abstract State）的值。状态抽象可以通过任何方法实现。

$$V(z) = (1 - \alpha_v) V(z) + \alpha_v \left( r_v + \gamma_v(z') \right) \tag{8}$$

$$F(s, s') = \gamma_v V(z') - V(z) \tag{9}$$

文献中的另一项工作是 [10]，提出了一种基于计划（Plan）的势函数。在 [10] 中，智能体根据在计划中的进展获得额外奖励。在公式 (10) 中，$z$ 是一个抽象状态，可以是计划的任何状态，函数 $\text{step}(z)$ 返回给定抽象状态 $z$ 在执行计划过程中出现的时间步。

$$\varphi(s) = \text{step}(z) \tag{10}$$

另一项工作是 [10] 在多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）中的扩展版本 [8]。在 [8] 中提出了两种方法，可用于塑形智能体的奖励信号并改善其学习过程。

文献中迁移学习领域的另一项工作是使用奖励塑形迁移智能体策略的方法。文献 [5] 的作者假设存在一个映射函数，将目标任务映射到源任务。利用该映射函数，他们定义了一个用于塑形奖励信号的势函数。如公式 (11) 所示，所提出的势函数基于源任务的策略定义。在公式 (11) 中，$X_S$ 是从目标任务状态空间到源任务状态空间的映射函数，$X_A$ 是从目标任务动作空间到源任务动作空间的映射函数，$\pi$ 是智能体在源任务中的策略。

$$\varphi(s, a) = \pi(X_S(s), X_A(a)) \tag{11}$$

在 [5] 中，RS 被用作知识迁移过程，从另一个任务中学习到的策略被用作知识。

文献中还有许多其他工作提出了用于多智能体强化学习和单智能体强化学习（Single-Agent Reinforcement Learning, SARL）的基于势函数的奖励塑形方法。[4] 提出了一种在存在任务演示时的 RS 方法，使用相似性度量来计算状态-动作对之间的相似性。另一项假设类似条件的工作是 [15]，其作者不使用相似性度量，而是使用逆强化学习（Inverse Reinforcement Learning）方法来近似用于构造势函数的奖励函数。[9] 提出了一种用于 MARL 的技术，使用差分奖励（Difference Reward）来构造势函数。差分奖励通过不考虑其他智能体动作对环境的影响来帮助智能体学习其自身动作对环境的影响。

---

## 4 所提出的基于势函数的奖励塑形方法

基于上述要点和动机，我们继续描述利用从学习过程中获得的知识来强化奖励信号的方法。因此，我们需要一个每当智能体在任务中取得进展时就会变化的奖励函数。该奖励函数必须根据目前为止最好和最差的回合来鼓励或惩罚智能体。奖励函数还必须能够处理稀疏奖励信号。公式 (12) 所表示的函数具有我们所期望的性质。正如 [14] 所述，改变奖励函数可能会改变当前任务的最优策略。因此，我们使用奖励塑形来操纵奖励函数，以考虑智能体在学习过程中的改进。

$$\varphi(s, a, t) = \begin{cases} 0 & R(s,a) = 0 \\ 1 + \dfrac{R_{ep} - R_{ep}^{u}(t)}{R_{ep}^{u}(t) - R_{ep}^{l}(t)} & \text{其他} \end{cases} \tag{12}$$

在公式 (12) 中：$R(s,a)$ 是即时奖励，为稀疏奖励信号；$R_{ep}$ 是当前回合的奖励总和，我们称之为回合奖励（Episode Reward）；$R_{ep}^{u}(t)$ 是截至当前的回合奖励最大值；$R_{ep}^{l}(t)$ 是截至当前的回合奖励最小值。该函数在环境奖励信号不包含信息时返回零，即控制稀疏奖励的影响。该函数通过测量与一个固定点的距离来强化奖励信号。我们可以将这种方法与任何类型的学习算法结合使用以提升学习过程。每个学习回合结束后，智能体将在需要时更改势函数的参数。

### 4.1 扩展到多任务智能体

所提出的势函数通过使用从先前回合中提取的知识帮助学习方法提高性能。有时智能体需要同时学习多个任务。考虑到这一点，我们将所提方法扩展为可用于多任务 RL。

---

## 5 实验结果

为了展示所提方法的有效性，我们在两个领域中进行了实验：街机学习环境（Arcade Learning Environment, ALE）[3] 中的 Breakout 和 Pong 游戏。根据 [3]：

> "ALE 提供了数百个 Atari 2600 游戏环境的接口，每个环境都各不相同、引人入胜，并被设计为对人类玩家具有挑战性。ALE 为强化学习、模型学习、基于模型的规划、模仿学习、迁移学习和内在动机提出了重要的研究挑战。"

**图 1**：单任务智能体在 Breakout 上的学习曲线。(a) 和 (b) 展示了不同条件下的学习曲线对比。

**图 2**：单任务智能体在 Pong 上的学习曲线。(a) 和 (b) 展示了不同条件下的学习曲线对比。

我们使用两个不同的基线来与所提方法进行比较。[1] 作为第一个基线，[6] 作为第二个基线。[1] 是 [13] 的实现，使用策略梯度方法来近似特定任务的最优策略。作为第二个基线，我们实现了迁移学习文献中提出的 [6]。我们在两个不同阶段评估我们的工作：首先在学习过程中评估我们的方法，然后评估最终策略的性能。我们在两种不同的假设下测试了我们的方法：第一种假设是我们已知每个任务中回合奖励的最大值和最小值；第二种假设是我们没有关于任务回合奖励的信息，这些值将在学习过程中获得。

### 5.1 Breakout

Breakout 是一款街机游戏，智能体控制一个挡板击球，试图摧毁更多砖块，同时防止球越过挡板。我们在该环境中训练智能体 1000000 个回合。图 1 展示了在 Breakout 游戏上训练的智能体的学习曲线。如图 1 所示，我们方法的曲线下面积大于基线方法，最终性能也优于基线。

### 5.2 Pong

Pong 也是一款街机游戏，智能体负责移动挡板击球。如果智能体丢球则获得 -1 奖励，如果对手丢球则获得 +1 奖励。我们在该环境中训练智能体 30000 个回合。图 2 展示了在 Pong 游戏中训练的智能体的学习曲线。如图 2 所示，我们方法的曲线下面积大于其中一个基线方法。然而，总体而言，所提方法在该环境中未能很好地工作，只有少量改进。

**图 3**：多任务智能体在 Pong 和 Breakout 上的学习曲线。(a) Pong 学习曲线，(b) Breakout 学习曲线。

**图 4**：多任务智能体在 Pong 和 Breakout 上的策略评估。(a) Pong 策略评估，(b) Breakout 策略评估。

### 5.3 多任务智能体

我们还实现了一个多任务智能体，学习如何玩 Pong 和 Breakout 游戏，然后训练智能体 115k 个回合。图 3 和图 4 展示了多任务智能体的实验结果。图 3 展示了多任务智能体在每个游戏上的学习曲线，图 4 展示了所学策略的性能。图 4 是将智能体以所学策略运行 100 个回合后取平均奖励的结果。可以看到，使用强化奖励信号的多任务智能体平均表现优于不使用强化奖励信号的多任务智能体。

---

## 6 结论

首先，本文研究了迁移学习和基于势函数的奖励塑形的文献。根据我们对迁移学习和基于势函数的奖励塑形文献的研究，没有方法尝试从学习过程本身提取知识。因此，我们引入了一种从学习过程中提取知识的新方法，并使用奖励塑形作为知识迁移方法。

所提方法通过观察回合奖励的值来引导智能体朝向目标。如果智能体朝目标前进，奖励信号将被强化。该方法可以与任何学习算法结合使用，并且预期在应用的任何场景中都有效。我们使用 [1] 实现了该方法，在两个不同环境中与两个不同基线进行了比较，然后将该方法扩展到多任务智能体中使用。在大多数实验中，结果令人鼓舞。我们还在一个同时学习 Pong 和 Breakout 游戏的多任务智能体中测试了我们的方法，发现该方法能够带来改进。

---

## 参考文献

[1] Mohammad Babaeizadeh, Iuri Frosio, Stephen Tyree, Jason Clemons, and Jan Kautz. Reinforcement learning thorugh asynchronous advantage actor-critic on a gpu. In 5th International Conference on Learning Representations, 2017.

[2] Peter L. Bartlett and Jonathan Baxter. Infinite-horizon policy-gradient estimation. Journal of Artificial Intelligence Research, 15(3):319–350, 2001.

[3] Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. J. Artif. Int. Res., 47(1):253–279, May 2013.

[4] Tim Brys, Anna Harutyunyan, Halit Bener Suay, Sonia Chernova, Matthew E. Taylor, and Ann Nowé. Reinforcement learning from demonstration through shaping. In Proceedings of the 24th International Conference on Artificial Intelligence, IJCAI'15, pages 3352–3358. AAAI Press, 2015.

[5] Tim Brys, Anna Harutyunyan, Matthew E. Taylor, and Ann Nowé. Policy transfer using reward shaping. In Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems, AAMAS '15, pages 181–188, Richland, SC, 2015. International Foundation for Autonomous Agents and Multiagent Systems.

[6] Gabriel De la Cruz, Yunshu Du, James Irwin, and Matthew Taylor. Initial progress in transfer for deep reinforcement learning algorithms. In International Joint Conference on Artificial Intelligence (IJCAI), 2016.

[7] Sam Devlin and Daniel Kudenko. Dynamic potential-based reward shaping. In Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems - Volume 1, AAMAS '12, pages 433–440, Richland, SC, 2012. International Foundation for Autonomous Agents and Multiagent Systems.

[8] Sam Devlin and Daniel Kudenko. Plan-based reward shaping for multi-agent reinforcement learning. Knowledge Eng. Review, 31(1):44–58, 2016.

[9] Sam Devlin, Logan Michael Yliniemi, Daniel Kudenko, and Kagan Tumer. Potential-based difference rewards for multiagent reinforcement learning. In AAMAS, pages 165–172. IFAAMAS/ACM, 2014.

[10] M. Grzes and D. Kudenko. Plan-based reward shaping for reinforcement learning. In 2008 4th International IEEE Conference Intelligent Systems, volume 2, pages 10–22–10–29, Sept 2008.

[11] Marek Grześ and Daniel Kudenko. Multigrid reinforcement learning with reward shaping. In Proceedings of the 18th International Conference on Artificial Neural Networks, Part I, ICANN '08, pages 357–366, Berlin, Heidelberg, 2008. Springer-Verlag.

[12] Anna Harutyunyan, Sam Devlin, Peter Vrancx, and Ann Nowe. Expressing arbitrary reward functions as potential-based advice. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, AAAI'15, pages 2652–2658. AAAI Press, 2015.

[13] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1928–1937, New York, New York, USA, 20–22 Jun 2016. PMLR.

[14] Andrew Y. Ng, Daishi Harada, and Stuart J. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the Sixteenth International Conference on Machine Learning, ICML '99, pages 278–287, San Francisco, CA, USA, 1999. Morgan Kaufmann Publishers Inc.

[15] Halit Bener Suay, Tim Brys, Matthew E. Taylor, and Sonia Chernova. Learning from demonstration for shaping through inverse reinforcement learning. In Proceedings of the 2016 International Conference on Autonomous Agents & Multiagent Systems, AAMAS '16, pages 429–437, Richland, SC, 2016. International Foundation for Autonomous Agents and Multiagent Systems.

[16] Richard S. Sutton and Andrew G. Barto. Introduction to Reinforcement Learning. MIT Press, Cambridge, MA, USA, 1st edition, 1998.

[17] Richard S. Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Proceedings of the 12th International Conference on Neural Information Processing Systems, NIPS'99, pages 1057–1063, Cambridge, MA, USA, 1999. MIT Press.

[18] Matthew E. Taylor and Peter Stone. Transfer Learning for Reinforcement Learning Domains: A Survey. JOURNAL OF MACHINE LEARNING RESEARCH, 10:1633–1685, JUL 2009.

[19] Christopher J.C.H. Watkins and Peter Dayan. Technical note: Q-learning. Machine Learning, 8(3):279–292, May 1992.

[20] Eric Wiewiora, Garrison Cottrell, and Charles Elkan. Principled methods for advising reinforcement learning agents. In Proceedings of the Twentieth International Conference on International Conference on Machine Learning, ICML'03, pages 792–799. AAAI Press, 2003.

[21] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, May 1992.
