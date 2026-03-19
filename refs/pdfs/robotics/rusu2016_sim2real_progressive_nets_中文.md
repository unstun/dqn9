# 基于渐进网络的从仿真到真实机器人像素级学习

**作者：** Andrei A. Rusu, Mel Vecerik, Thomas Rothorl, Nicolas Heess, Razvan Pascanu, Raia Hadsell

**单位：** DeepMind, London, UK

**发表：** 1st Conference on Robot Learning (CoRL 2017)

**arXiv:** 1610.04286v2 [cs.RO] 22 May 2018

---

## 摘要

将端到端学习（end-to-end learning）应用于机器人上基于像素的复杂交互控制任务仍是一个未解决的问题。深度强化学习（Deep Reinforcement Learning）算法在真实机器人上训练速度过慢，但其潜力已在仿真环境中得到验证。我们提出使用渐进网络（Progressive Networks）来弥合现实差距（reality gap），将从仿真中学习到的策略迁移到真实世界。渐进网络方法是一个通用框架，能够复用从低层视觉特征到高层策略的所有内容，以实现向新任务的迁移，从而以组合式且简洁的方式构建复杂技能。我们通过一系列机器人操作领域的实验展示了该方法的早期成果，重点在于弥合现实差距。与其他已提出的方法不同，我们的真实世界实验证明了在全驱动机械臂上从原始视觉输入成功学习任务的能力。此外，任务学习仅使用深度强化学习和稀疏奖励（sparse rewards）完成，而非依赖基于模型的轨迹优化。

**关键词：** 机器人学习、迁移、渐进网络、仿真到真实、CoRL

---

## 1 引言

深度强化学习为在机器人领域实现人类水平的控制提供了新的希望，尤其是在像素到动作（pixel-to-action）的场景中，状态估计来自高维传感器，环境交互和反馈至关重要。借助深度强化学习，一系列新算法能够在具有挑战性的任务上实现精密的控制，但这些成果主要在仿真中得到验证，而非在实际机器人平台上。

尽管仿真驱动的深度强化学习近期取得了令人瞩目的进展 [1, 2, 3, 4, 5, 6, 7]，在真实机器人上展示学习能力仍然是衡量这些方法实际适用性的标准。然而这构成了巨大挑战，因为当前基于像素的深度强化学习方法需要"数据密集型"训练机制，且研究用机器人及其操作人员相对脆弱。一种解决方案是使用迁移学习（transfer learning）方法来弥合仿真与真实世界之间的现实差距。在本文中，我们使用渐进网络——一种近期提出的用于迁移学习的深度学习架构——来展示这种方法，从而提供一条概念验证路径，使深度强化学习能够在真实机器人上实现快速策略学习。

渐进网络已被证明能够通过与先前学习模型的横向连接（lateral connections），在 Atari 游戏等差异较大的任务之间实现正迁移 [8]。为每个新任务添加新容量允许学习专门的输入特征，这对于受益于精细调优感知特征的深度强化学习算法来说是一个重要优势。与其他迁移学习或领域自适应（domain adaptation）方法相比，渐进网络的一个优势在于可以顺序学习多个任务，而无需指定源任务和目标任务。

本文提出了一种从仿真到真实机器人的迁移方法，并通过真实世界的稀疏奖励任务进行了验证。任务使用端到端深度强化学习进行学习，输入为 RGB 图像，输出为关节速度动作。首先，使用多个异步工作进程在仿真中训练一个 Actor-Critic 网络 [6]。该网络包含一个卷积编码器和一个 LSTM。从 LSTM 状态出发，通过线性层计算一组离散动作输出，用于控制仿真机器人的各自由度以及值函数。训练完成后，初始化一个新网络，通过横向非线性连接与仿真训练网络的每个卷积层和循环层相连。新网络在真实机器人上的类似任务中进行训练。我们的初步发现表明，仿真网络的特征和编码策略所赋予的归纳偏置（inductive bias）足以在真实机器人上带来显著的学习加速。

---

## 2 从仿真到真实的迁移学习

我们的方法依赖于渐进网络架构，该架构通过横向连接实现迁移学习，将先前学习的网络列（column）的每一层连接到每个新列，从而支持丰富的特征组合。我们首先概述渐进网络，然后讨论其在机器人领域迁移中的应用。

### 2.1 渐进网络

渐进网络非常适合机器人控制领域中从仿真到真实的策略迁移，原因有多个。第一，为某一任务学习的特征可以迁移到许多新任务中，而不会因微调（fine-tuning）而被破坏。第二，各列可以是异构的，这对于解决不同任务、处理不同输入模态或单纯提高迁移到真实机器人时的学习速度可能很重要。第三，渐进网络在迁移到新任务时会添加新的容量，包括新的输入连接。这有利于弥合现实差距，以适应仿真与真实传感器之间不同的输入。

渐进网络以单一列开始：一个具有 $L$ 层的深度神经网络，隐藏激活为 $h^{(1)}_i \in \mathbb{R}^{n_i}$，其中 $n_i$ 为第 $i \leq L$ 层的单元数，参数 $\Theta^{(1)}$ 训练至收敛。当切换到第二个任务时，参数 $\Theta^{(1)}$ 被"冻结"，并实例化一个具有参数 $\Theta^{(2)}$ 的新列（随机初始化），其中第 $h^{(2)}_i$ 层通过横向连接同时接收来自 $h^{(2)}_{i-1}$ 和 $h^{(1)}_{i-1}$ 的输入。渐进网络可以直接推广到每列/每层具有任意网络宽度，以适应不同的任务难度，或者在集成设置中编译来自多个独立网络的横向连接。

$$h^{(k)}_i = f\left(W^{(k)}_i h^{(k)}_{i-1} + \sum_{j<k} U^{(k:j)}_i h^{(j)}_{i-1}\right) \tag{1}$$

其中 $W^{(k)}_i \in \mathbb{R}^{n_i \times n_{i-1}}$ 是第 $k$ 列第 $i$ 层的权重矩阵，$U^{(k:j)}_i \in \mathbb{R}^{n_i \times n_j}$ 是从第 $j$ 列第 $i-1$ 层到第 $k$ 列第 $i$ 层的横向连接，$h_0$ 是网络输入。$f$ 是逐元素非线性函数：对所有中间层使用 $f(x) = \max(0, x)$。

在标准的预训练加微调范式中，通常隐含假设任务之间存在"重叠"。在这种情况下微调是高效的，因为参数只需对目标域进行轻微调整，通常只需重新训练顶层。相比之下，我们不对任务之间的关系做任何假设，任务在实践中可能是正交的甚至是对抗性的。渐进网络通过为每个新任务分配一个新列来规避这个问题，新列可能具有不同的结构或输入。渐进网络中的各列可以通过横向连接自由地复用、修改或忽略先前学习的特征。

**在强化学习中的应用。** 尽管渐进网络具有广泛的适用性，本文重点关注其在深度强化学习中的应用。在这种情况下，每一列被训练来解决一个特定的马尔可夫决策过程（Markov Decision Process, MDP）：第 $k$ 列因此定义了一个策略 $\pi^{(k)}(a|s)$，以环境给出的状态 $s$ 作为输入，生成动作的概率分布 $\pi^{(k)}(a|s) := h^{(k)}_L(s)$。在每个时间步，从该分布中采样一个动作并在环境中执行，产生后续状态。该策略隐式地定义了状态和动作上的平稳分布 $\rho_{\pi^{(k)}}(s, a)$。

### 2.2 方法

所提出的从仿真到真实机器人领域的迁移方法基于渐进网络，并进行了一些特定的修改。第一，渐进网络的各列不需要具有相同的容量或结构，这在仿真到真实的情况下可以是一个优势。因此，仿真训练的列被设计为具有足够的容量和深度以从零开始学习任务，但机器人训练的列具有最小容量，以鼓励快速学习并限制总参数增长。第二，为渐进网络提出的逐层适配器对于互补任务序列的输出层是不必要的，因此不使用它们。第三，机器人训练列的输出层从仿真训练列初始化，以改善探索。这些架构特征如图 1 所示。

> **图 1：** 左侧为渐进网络示意图，右侧为用于机器人迁移学习的改进渐进架构。第一列在仿真中对任务 1 进行训练，第二列在机器人上对任务 1 进行训练，第三列在机器人上对任务 2 进行训练。各列容量可以不同，适配器函数（标记为 'a'）不用于该非对抗性任务序列的输出层。

在这种迁移学习方法中，最大的风险是真实域中的奖励过于稀疏或不存在，导致强化学习无法在实际时间范围内改善一个严重次优的初始策略。因此，为了在真实域的探索中最大化获得奖励的可能性，新列的初始化方式使得智能体的初始策略与前一列相同。具体实现方式为：将前一列最后一层到新列输出层的权重用前一列的输出权重初始化，而来自当前列最后隐藏层的连接权重初始化为零。因此，以图 1（右）中的示例网络为例，当实例化参数 $\Theta^{(2)}$ 时，层 $\text{output}^{(2)}_2$ 将接收来自 $h^{(1)}_2$ 和 $h^{(2)}_2$ 的输入连接。然而，与 $\Theta^{(2)}$ 中其他随机初始化的参数不同，权重 $W^{(2)}_{\text{out}}$ 将为零，权重 $U^{(1:2)}_{\text{out}}$ 将从 $W^{(1)}_{\text{out}}$ 复制而来。注意这仅影响智能体的初始策略，并不阻止新列的训练。

---

## 3 相关工作

领域迁移（domain transfer）存在许多不同的范式，也有许多专门为深度神经模型设计的方法，但专门用于机器人领域从仿真到真实迁移的方法要少得多。能够用于交互式、丰富传感器域中端到端（像素到动作）学习迁移的方法更是罕见。

越来越多的工作在研究深度网络的跨域迁移能力。一些研究 [9, 10] 考虑在存在对齐的情况下，简单地用源域数据增强目标域数据。在此基础上，文献 [11] 从以下观察出发：当观察模型中更高的层时，特征的可迁移性迅速下降。为了纠正这一效应，添加了一个软约束来强制特征分布更加相似。文献 [11] 提出了一种"混淆"损失（confusion loss），迫使模型忽略区分两个域的数据变化 [12, 13]。

基于文献 [12]，文献 [14] 尝试使用对齐数据来解决仿真到真实的差距。该工作聚焦于机械臂的姿态估计，训练采用三元组损失（triple loss），考察对齐的仿真到真实数据，包括域混淆损失。该论文未展示该方法在学习新的复杂策略方面的效率。

监督学习文献中的几项近期工作，如 [15, 16, 17]，展示了如何利用神经网络对抗训练（adversarial training）的思想来降低训练网络对域间差异的敏感性，而无需对齐的训练数据。直觉上，这些方法训练一种表示，使得难以区分来自不同域的数据点。然而，这些思想尚未在控制领域中进行测试。

展示了该问题的难度，文献 [10] 提供了证据表明，简单地将在合成数据上训练的模型应用于真实机器人会失败。该论文还表明，主要的失败点在于仿真与真实之间视觉线索的差异。

从仿真到真实机器人的部分成功迁移已有报道 [18, 19, 20]。这些工作主要关注从更受限的简单版本任务向完整的更困难版本任务的迁移问题。虽然从仿真到真实的迁移仍然困难，但在真实机器人上直接学习神经网络控制策略方面已取得进展，包括从低维状态表示和视觉输入学习，如 [21, 22]。虽然这些结果令人印象深刻，但为了实现足够的数据效率，这些工作目前依赖于相对受限的任务设置、专门的视觉架构和精心设计的训练机制。替代方法则将大数据思想引入机器人学 [23, 24]。

---

## 4 实验

在仿真中的训练使用文献 [6] 中引入的异步优势 Actor-Critic（Asynchronous Advantage Actor-Critic, A3C）框架。与 DQN [25] 相比，该模型同时学习策略和用于预测期望未来奖励的值函数，并且可以使用 CPU 通过多线程进行训练。A3C 已被证明比 DQN 收敛更快，这使其在研究实验中具有优势。

对于 Jaco 机械臂的操作领域，智能体策略使用速度命令控制九个自由度，包括手臂上的六个关节和三个驱动手指。完整策略 $\Pi(A|s, \theta)$ 包含智能体学习的九个关节策略，每个都是连接到前一层和任何横向连接输入的 softmax。每个关节策略 $i$ 有三个动作（固定正速度、固定负速度和零速度）：$\pi_i(a_i|s; \theta_i)$。这种离散动作集虽然可能缺乏连续控制策略的精度，但在实践中表现良好。还有一个单一的值函数，线性连接到前一层和横向层：$V(s, \theta_v)$。

我们评估了前馈网络和循环神经网络。两者都有卷积输入层，后接全连接层或 LSTM。仿真训练列使用标准大小的网络，机器人训练列使用缩减容量的网络，选择缩减容量是因为我们实验发现更多容量并不会加速学习（见第 4.2 节），这可能是因为复用了前一列的特征。架构细节如图 2 和表 1 所示。在所有变体中，输入为 3x64x64 像素，输出为 28（9 个离散关节策略加一个值函数）。

MuJoCo 物理仿真器 [26] 用于训练我们实验中的第一列，使用渲染的摄像机视图提供观测。在真实域中，一个位置类似的 RGB 摄像机提供输入。虽然建模的 Jaco 及其动力学相当准确，但视觉差异是明显的，如图 3 所示。

> **图 2：** 渐进循环网络架构的详细示意图。LSTM 的激活作为输入连接到渐进列。图中展示了分解策略和单一值函数。

> **表 1：** 宽列（仿真训练）和窄列（机器人训练）的网络大小。对于所有网络，第一个卷积层使用 8x8、步长 4 的卷积核，第二个使用 5x5、步长 2 的卷积核。总参数包括横向连接。
>
> | | 前馈-宽 | 前馈-窄 | 循环-宽 | 循环-窄 |
> |---|---|---|---|---|
> | fc (输出) | 28 | 28 | 28 | 28 |
> | LSTM | - | - | 128 | 16 |
> | fc | 512 | 32 | 128 | 16 |
> | conv 2 | 32 | 8 | 32 | 8 |
> | conv 1 | 16 | 8 | 16 | 8 |
> | 参数量 | 621K | 39K | 299K | 37K |

> **图 3：** 真实摄像机输入图像和 MuJoCo 渲染图像的示例。虽然可以使用更逼真的模型外观，但为了加速在 CPU 上进行的 MuJoCo 渲染，使用了块状的 Jaco 模型。图像展示了 Jaco 起始位置和目标位置的多样性。

所有实验都围绕到达视觉目标的任务展开，仅提供纯奖励作为反馈（无塑形奖励）。虽然简单，但该任务要求从视觉观测中正确推断手臂状态和目标位置，并且智能体需要学习在高维状态空间上的鲁棒控制。手臂在每个回合（episode）开始时被设置为随机起始位置，目标在 40cm x 30cm 的区域内随机放置。如果手掌距离目标 10cm 以内，智能体获得 +1 的奖励，每个回合最多持续 50 步。尽管由于随机起始状态存在一些方差，表现良好的智能体可以通过快速到达目标并始终保持在安全位置来获得超过 30 分的平均得分。如果智能体因自交叉、触碰桌面或超出设定的关节限制而导致安全违规，则回合终止。

### 4.1 仿真中的训练

第一列使用 A3C 在仿真中训练，如前所述，使用宽型前馈或循环网络。直觉上，在仿真中使用更大容量的网络以达到最大性能是合理的。我们通过比较宽型和窄型网络架构验证了这一直觉，发现窄型网络学习较慢且性能较差（见图 4）。我们还发现 LSTM 模型比前馈模型平均每回合高出 3 分。即使在这个相对简单的任务上，也需要与环境进行大量交互（约 5000 万步）才能达到完全性能——这一数字在真实机器人上是不可行的。

> **图 4：** 展示了前馈（左）和循环（右）模型的宽型和窄型版本在 MuJoCo 仿真器中训练的学习曲线。图中显示了 5 次不同种子和超参数训练运行的均值和方差。稳定性能在约 5000 万步后达到，即超过 100 万个回合。虽然前馈和循环模型都能学习该任务，但循环网络达到了更高的最终平均分。

与真实机器人相比，仿真训练的加速来自快速渲染、多线程学习算法以及无需人工干预即可持续训练的能力。我们计算得出，学习该任务在 CPU 计算集群上训练至收敛需要 24 小时，而在真实机器人上即使全天 24 小时连续训练也需要 53 天。此外，在仿真中使用了多个并行实验来探索超参数；这种搜索将使假设的真实机器人训练时间成倍增加。

在仿真中，我们探索学习率和熵代价（entropy cost），它们在对数尺度上均匀随机采样。学习率在 5e-5 和 5e-3 之间采样，熵代价在 1e-5 和 1e-2 之间采样。从 30 个网格中选择最终性能最佳的配置作为第一列。对于真实 Jaco 实验，使用单线程智能体（A2C）的仿真迁移实验分别优化学习率和熵代价。

### 4.2 迁移到机器人

为了在真实 Jaco 上训练，每三个回合手动重新放置一个平面目标，范围在 40cm x 30cm 的区域内。通过跟踪彩色目标并根据 Jaco 夹爪相对于目标的位置给出奖励来自动提供奖励。我们训练了从零开始的基线、微调的第一列和渐进的第二列。每个实验运行约 60000 步（约四小时）。基线通过随机初始化窄型网络进行训练。我们还尝试了随机初始化的宽型网络。如图 5（绿色曲线）所示，随机初始化的列无法学习，智能体在整个训练过程中获得零奖励。渐进的第二列达到 34 分，而从仿真训练列开始在机器人上继续训练的微调实验未能达到渐进网络的相同得分。

> **图 5：** 真实机器人训练：比较渐进方法、微调方法和"从零开始"的学习曲线。所有实验使用循环架构，在机器人上从 RGB 输入进行训练。对渐进实验和随机初始化基线分别比较了宽列和窄列。对于所有结果，中值滤波的实线叠加在原始奖励（虚线）上。"从零开始"基线是随机初始化的窄列或宽列，两者在训练期间都未能获得任何奖励。

**微调与渐进方法的对比。** 渐进方法显然非常适合持续学习（continual learning）场景，在这些场景中减轻对先前任务的遗忘同时支持向新任务的迁移很重要，但对于重点在于最大化迁移学习的任务课程，其优势不那么直观。为了进行实证评估，我们从上述仿真训练的第一列开始，然后在多种条件下（包括小幅或大幅颜色变化和小幅或大幅视角变化）对该列进行微调或添加窄型渐进列并重新训练到达任务。对于每种环境扰动，我们使用不同的种子、学习率和熵代价训练 300 次，这些是最敏感的超参数。如图 6 所示，我们发现渐进网络比微调更稳定，且达到更高的最终性能。

> **图 6：** 为分析微调与渐进方法的相对稳定性和性能，我们在仿真中向环境添加颜色或视角变化，然后使用不同的随机种子、学习率和熵代价训练 300 个网络。在全部四个实验中，渐进网络表现出显著更高的性能和更低的超参数选择敏感性。

### 4.3 迁移到带本体感知的动态机器人任务

与无法适应网络形态变化或新输入模态的微调范式不同，渐进网络提供了一种灵活性，有利于在利用先前知识的同时迁移到新的数据源。为了证明这一点，我们在到达任务上训练第二列，但添加本体感知（proprioception）特征作为与 RGB 图像并行的额外输入。本体感知特征为手臂和手指 9 个关节的关节角度和速度，共 18 个，输入到一个多层感知机 MLP（单个线性层加 ReLU），并与卷积堆叠的输出合并。然后，添加第三个渐进列，仅从本体感知特征学习，而视觉输入通过前面的列前向传递，特征通过横向连接使用。该架构的示意图如图 7（左）所示。

为了评估该架构，我们在动态目标任务上进行训练。通过使用小型电动滑轮，红色目标在桌面上平滑移动并随机反向运动，创建了一个需要不同控制策略同时保持类似视觉呈现的跟踪任务。任务的其他方面，包括奖励和回合长度，保持不变。如果第二列在此传送带任务上训练，学习相对较慢，完全性能在 50000 步（约 4 小时）后达到。如果第二列在静态到达任务上训练，然后第三列在传送带任务上训练，我们观察到即时迁移，完全性能几乎立即达到（图 7 右）。这证明了渐进网络在课程任务中的实用性，以及该架构立即复用先前学习特征的能力。

> **图 7：** 展示了动态"传送带"任务的真实机器人训练结果。左侧描绘了三列架构，其中视觉 (x) 用于训练第一列，视觉和本体感知 (phi) 用于第二列，仅本体感知用于训练第三列。编码器 1 是卷积网络，编码器 2 是在 LSTM 之前添加了本体感知特征的卷积网络，编码器 3 是 MLP。右侧学习曲线展示了在传送带（动态目标）任务上的训练结果。如果传送带任务作为第三列而非第二列学习，则学习显著更快。

---

## 5 讨论

迁移学习——积累和迁移知识到新领域的能力——是智能体的核心特征。渐进神经网络提供了一个框架，可用于多任务的持续学习，并促进迁移学习，甚至跨越仿真与机器人之间的鸿沟。我们充分利用了仿真提供的灵活性和计算扩展能力，比较了许多超参数和架构用于具有视觉输入的随机起始、随机目标控制任务，然后成功地将技能迁移到在真实机器人上训练的智能体。

为了发挥深度强化学习在真实世界机器人领域的潜力，学习效率需要提高许多倍。实现这一目标的一条途径是通过从仿真训练的智能体进行迁移学习。我们描述了一组初步实验，证明渐进网络可用于实现可靠、快速的像素到动作强化学习策略迁移。

---

## 参考文献

[1] S. Levine and P. Abbeel. Learning neural network policies with guided policy search under unknown dynamics. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 27, pages 1071–1079. Curran Associates, Inc., 2014.

[2] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015.

[3] N. Heess, G. Wayne, D. Silver, T. P. Lillicrap, T. Erez, and Y. Tassa. Learning continuous control policies by stochastic value gradients. In Advances in Neural Information Processing Systems 28, 2015.

[4] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning. Proceedings of the International Conference on Learning Representations (ICLR), 2016.

[5] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

[6] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Int'l Conf. on Machine Learning (ICML), 2016.

[7] S. Gu, T. P. Lillicrap, I. Sutskever, and S. Levine. Continuous deep q-learning with model-based acceleration. In ICML 2016, 2016.

[8] A. Rusu, N. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

[9] X. Peng, B. Sun, K. Ali, and K. Saenko. Learning deep object detectors from 3d models. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, pages 1278–1286, 2015.

[10] H. Su, C. R. Qi, Y. Li, and L. J. Guibas. Render for CNN: viewpoint estimation in images using cnns trained with rendered 3d model views. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, pages 2686–2694, 2015.

[11] M. Long, Y. Cao, J. Wang, and M. I. Jordan. Learning transferable features with deep adaptation networks. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, pages 97–105, 2015.

[12] E. Tzeng, J. Hoffman, T. Darrell, and K. Saenko. Simultaneous deep transfer across domains and tasks. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, pages 4068–4076, 2015.

[13] E. Tzeng, J. Hoffman, N. Zhang, K. Saenko, and T. Darrell. Deep domain confusion: Maximizing for domain invariance. CoRR, abs/1412.3474, 2014.

[14] E. Tzeng, C. Devin, J. Hoffman, C. Finn, X. Peng, S. Levine, K. Saenko, and T. Darrell. Towards adapting deep visuomotor representations from simulated to real environments. CoRR, abs/1511.07111, 2015.

[15] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(59):1–35, 2016.

[16] H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, and M. Marchand. Domain-adversarial neural networks. CoRR, abs/1412.4446, 2014.

[17] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan. Domain separation networks. In Advances in Neural Information Processing Systems, pages 343–351, 2016.

[18] S. Barrett, M. E. Taylor, and P. Stone. Transfer learning for reinforcement learning on a physical robot. In Ninth International Conference on Autonomous Agents and Multiagent Systems - Adaptive Learning Agents Workshop (AAMAS - ALA), 2010.

[19] S. James and E. Johns. 3D Simulation for Robot Arm Control with Deep Q-Learning. ArXiv e-prints, 2016.

[20] Y. Zhu, R. Mottaghi, E. Kolve, J. J. Lim, A. Gupta, L. Fei-Fei, and A. Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. In Robotics and Automation (ICRA), 2017 IEEE International Conference on, pages 3357–3364. IEEE, 2017.

[21] S. Levine, C. Finn, T. Darrell, and P. Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17(39):1–40, 2016.

[22] S. Levine, N. Wagener, and P. Abbeel. Learning contact-rich manipulation skills with guided policy search. In IEEE International Conference on Robotics and Automation, ICRA 2015, pages 156–163, 2015.

[23] L. Pinto and A. Gupta. Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours. In ICRA 2016, 2016.

[24] S. Levine, P. Pastor, A. Krizhevsky, J. Ibarz, and D. Quillen. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International Journal of Robotics Research, page 0278364917710318, 2016.

[25] V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

[26] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In International Conference on Intelligent Robots and Systems IROS, 2012.
