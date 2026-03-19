# 一天内学会驾驶

**Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley, Amar Shah**

*作者均来自英国剑桥 Wayve 公司。research@wayve.ai*

---

## 摘要

我们展示了深度强化学习（deep reinforcement learning）在自动驾驶（autonomous driving）中的首次应用。从随机初始化的参数出发，我们的模型能够仅使用单目图像（monocular image）作为输入，在少量训练回合（training episodes）内学会车道跟随（lane following）策略。我们提供了一种通用且易于获取的奖励信号：即车辆在安全驾驶员接管控制之前行驶的距离。我们使用连续动作空间的无模型深度强化学习算法（continuous, model-free deep reinforcement learning algorithm），所有探索和优化均在车辆上执行。这展示了一种新的自动驾驶框架，摆脱了对预定义逻辑规则、地图和直接监督的依赖。我们讨论了将该方法扩展到更广泛自动驾驶任务的挑战和机遇。

---

## I. 引言

自动驾驶是一个引起学术界和工业界广泛关注的话题，因其具有从根本上改变出行和交通的潜力。目前大多数方法主要依赖形式化逻辑，在标注的三维几何地图中定义驾驶行为。这种方法难以扩展，因为它严重依赖外部地图基础设施，而非主要基于对局部场景的理解。

为了使自动驾驶成为真正的通用技术，我们倡导构建能够在无地图和无显式规则条件下驾驶和导航的机器人系统——就像人类一样——依赖于对即时环境的全面理解 [1]，同时遵循简单的高层方向指令（如逐弯路线命令）。该领域的近期工作表明，利用 GPS 进行粗定位以及 LIDAR 理解局部场景，这在乡村道路上是可行的 [2]。

近年来，强化学习（reinforcement learning, RL）——一个专注于求解马尔可夫决策过程（Markov Decision Problems, MDP）[3] 的机器学习子领域，其中智能体通过在环境中选择动作来最大化某种奖励函数——已展示出在围棋 [4] 或国际象棋 [5] 等游戏中达到超人水平的能力，在计算机游戏等仿真环境中表现出巨大潜力 [6]，以及在简单的机器人操作任务中取得了成功 [7]。我们认为强化学习的通用性使其成为应用于自动驾驶的有用框架。最重要的是，它提供了一种纠正机制来改进所学习的自动驾驶行为。

为此，在本文中我们：

1. 将自动驾驶建模为 MDP，阐述如何设计该问题的各个要素使其更易求解，同时保持通用性和可扩展性；
2. 展示一种经典 RL 算法——深度确定性策略梯度（deep deterministic policy gradients, DDPG）[8]——能够在仿真环境中快速学会简单的自动驾驶任务；
3. 讨论在真实车辆上高效且安全地进行驾驶学习所需的系统配置；
4. 使用连续动作空间的深度强化学习算法，仅利用车载计算，在少量回合内学会驾驶真实的自动驾驶车辆。

因此，我们展示了深度强化学习智能体驾驶真实汽车的首次演示。

> **图 1**：我们设计了用于自动驾驶的深度强化学习算法。该图展示了我们用来学习驾驶策略和价值函数的演员-评论家算法（actor-critic algorithm）。我们的智能体最大化安全驾驶员干预前行驶距离作为奖励。图中描绘了网络架构：演员网络（Actor）包含卷积层和全连接层，输出转向和速度命令；评论家网络（Critic）接收状态-动作向量，输出 Q 值。车辆学习驾驶的视频见 https://wayve.ai/blog/l2diad

---

## II. 相关工作

我们相信这是首个展示深度强化学习可作为自动驾驶可行方法的工作。我们受到其超越模仿学习扩展潜力的激励，并希望研究社区能更深入地从强化学习角度研究自动驾驶。当前文献中最相关的工作主要可分为模仿学习和依赖地图的经典方法两类。

### a) 基于地图的方法

自早期示例 [9]、[10] 以来，自动驾驶系统被设计为使用先进的传感和控制算法在复杂环境中安全导航 [11]、[12]、[13]。这些系统传统上由许多独立工程化的特定组件组成，如感知、状态估计、建图、规划和控制 [14]。然而，由于每个组件都需要单独指定和调优，复杂的相互依赖关系使其难以扩展到更困难的驾驶场景。

大量工作聚焦于这种模块化方法中的计算机视觉组件。定位方法如 [15] 有助于在映射环境中控制车辆 [16]，而语义分割（semantic segmentation）等感知方法 [1] 使机器人能够理解场景。这些模块化任务由基准测试如 [17] 和 [18] 提供支持。

这种模块化地图方法是商业化自动驾驶系统开发的主要方向；然而，它们带来了极其复杂的系统工程挑战，至今尚未解决。

### b) 模仿学习

一种用于某些驾驶任务的较新方法是模仿学习（imitation learning）[19]、[20]，旨在通过观察专家演示来学习控制策略。该方法的一个重要优势是可以使用端到端深度学习（end-to-end deep learning），针对最终目标联合优化模型的所有参数，从而减少各组件调优的工作量。然而，模仿学习同样难以扩展。对于智能体可能遇到的每种潜在场景，不可能获得用于模仿的专家示例，并且处理演示策略的分布（如在不同车道中驾驶）也具有挑战性。

### c) 强化学习

强化学习是求解马尔可夫决策过程（MDPs）[21] 的一大类算法。一个 MDP 由以下部分组成：

- 状态集 $S$
- 动作集 $A$
- 转移概率函数 $p: S \times A \to \mathcal{P}(S)$，对每一对 $(s, a) \in S \times A$ 赋予一个概率分布 $p(\cdot|s, a)$，表示在状态 $s$ 下使用动作 $a$ 进入某状态的概率
- 奖励函数 $R: S \times S \times A \to \mathbb{R}$，描述从状态 $s_t$ 使用动作 $a_t$ 进入状态 $s_{t+1}$ 所获得的奖励 $R(s_{t+1}, s_t, a_t)$
- 未来折扣因子 $\gamma \in [0, 1]$，表示我们对未来奖励的重视程度

MDP 的解是一个策略 $\pi: S \to A$，对所有 $s_0 \in S$ 最大化：

$$V^\pi(s_0) = \mathbb{E}\left(\sum_{t=0}^{\infty} \gamma^t R(s_{t+1}, s_t, \pi(s_t))\right), \tag{1}$$

其中期望是对按 $p(s_{t+1}|s_t, \pi(s_t))$ 采样的状态 $s_{t+1}$ 取的。

在我们的设置中，使用有限时间范围 $T$ 代替上式中的无穷大。这等价于某个状态为终止状态（terminal），即无法离开该状态，且在该状态下任何动作的奖励为零。

将上述等式重新排列为递归形式，得到两个贝尔曼方程（Bellman equations）之一：

$$V^\pi(s_0) = \mathbb{E}\left(R(s_1, s_0, \pi(s_0)) + \gamma V^\pi(s_1)\right). \tag{2}$$

其中期望仅对按 $p(s_1|s_0, \pi(s_0))$ 采样的 $s_1$ 取。作为参考，给出另一个贝尔曼方程：

$$Q^\pi(s_0, a_0) = \mathbb{E}\left(R(s_1, s_0, a_0) + \gamma Q^\pi(s_1, \pi(s_1))\right), \tag{3}$$

其中 $Q^\pi(s_0, a_0)$ 是从状态 $s_0$ 采取动作 $a_0$ 并此后遵循策略 $\pi$ 所获得的期望累积折扣奖励。同样，期望是对按 $p(s_1|s_0, a_0)$ 采样的 $s_1$ 取的。

换言之，强化学习算法旨在学习一个能获得高累积奖励的策略 $\pi$。它们通常分为两类：基于模型的强化学习（model-based）和无模型的强化学习（model-free）。前者学习转移函数和奖励函数的显式模型，然后用其找到在估计函数下最大化累积奖励的策略。后者直接估计在状态 $s$ 下采取动作 $a$ 的价值 $Q(s, a)$，然后遵循在每个状态选择最高估计价值动作的策略。

无模型强化学习具有极高的通用性。使用它，我们（理论上）可以学习任何能想象到的任务，而基于模型的算法的性能只能与所学模型一样好。另一方面，基于模型的方法往往比无模型方法更具数据效率。进一步讨论参见 [22]。

在自动驾驶中，深度学习已被用于使用离线数据为基于模型的强化学习学习动力学模型 [23]。强化学习还被用于在视频游戏中学习自动驾驶智能体。然而，这可以简化问题，因为可以获得真实世界中不可用的真值奖励信号，如汽车相对车道的角度 [8]。

与本文最接近的工作来自 Riedmiller 等人 [24]，他们训练了一个强化学习智能体，使车辆在无障碍环境中沿 GPS 轨迹行驶。他们展示了在车辆上使用基于 GPS 阈值跟踪误差的密集奖励函数进行学习。我们在多方面扩展了这项工作：我们展示了使用深度学习从基于图像的输入、使用稀疏奖励函数进行车道跟随的驾驶学习。

---

## III. 系统架构

### A. 将驾驶建模为马尔可夫决策过程

本文的一个关键重点是将驾驶设置为 MDP。我们的目标是自动驾驶，状态空间 $S$、动作空间 $A$ 和奖励函数 $R$ 的确切定义由我们自由确定。一旦确定了状态和动作表示，转移模型就隐式固定了，剩余的自由度——转移本身——由所使用的仿真器/车辆的力学特性决定。

#### a) 状态空间

定义状态空间的关键是定义算法在每个时间步接收的观测 $O_t$。许多传感器被开发出来为驾驶算法提供复杂的观测，包括 LIDAR、IMU、GPS 单元和红外深度传感器等；高级传感技术可以消耗无尽的预算。在本文中，我们表明对于简单的驾驶任务，使用单目相机图像加上观测到的车辆速度和转向角就足够了。

理论上，状态 $s_t$ 应该是所有先前观测的马尔可夫表示。可以通过例如使用循环神经网络（Recurrent Neural Network）递归组合观测来获得固定长度的近似马尔可夫状态。然而，对于我们考虑的任务，观测本身就是状态的足够好的近似。

第二个考虑因素是如何处理图像本身：原始图像可以通过一系列卷积直接馈入强化学习算法 [25]；或者可以使用例如变分自编码器（Variational Autoencoder, VAE）[26]、[27] 提供图像的小型压缩表示。我们在第 IV 节比较了使用这两种方法的强化学习性能。在实验中，我们使用 KL 损失和 L2 重构损失 [27]，从五个纯随机探索回合中在线训练 VAE。

#### b) 动作空间

驾驶本身具有一组看似自然的动作：油门、刹车、转向信号等。但强化学习算法的输出域应该是什么？油门本身可以描述为离散的（开或关）或连续的（在等距于 $[0, 1]$ 的范围内）。另一种替代方案是将油门重新参数化为速度设定点（speed set-point），由经典控制器输出油门以匹配设定点。总体而言，在简单仿真器上的实验（第 IV-A 节）表明，连续动作虽然学习难度略高，但能提供更平滑的控制器。我们使用二维动作空间：转向角在 $[-1, 1]$ 范围内，速度设定点以 km/h 为单位。

#### c) 奖励函数

奖励函数设计可以接近监督学习——给定车道分类系统，可以将车道跟随的奖励设置为最小化预测的与车道中心距离，这是 [8] 中采用的方法。这种方法在扩展性上有限：系统的性能只能与人工制定奖励背后的人类直觉一样好。我们不采用这种方法。相反，我们将奖励定义为前进速度，并在违反交通规则时终止一个回合——因此给定状态的价值 $V(s_t)$ 对应于违规前平均行驶的距离。一个可能被识别的缺陷是，智能体可能选择避免更困难的操作，例如在英国右转（在美国为左转）。条件化命令的奖励可能在未来工作中被使用以避免这一问题。

### B. 强化学习算法——深度确定性策略梯度

我们选择了一种简单的连续动作域无模型强化学习算法：深度确定性策略梯度（DDPG）[8]，以表明一个现成的强化学习算法无需针对特定任务的适配即可求解第 III-A 节中提出的 MDP。

DDPG 由两个函数近似器组成：一个评论家 $Q: S \times A \to \mathbb{R}$，估计在状态 $s$ 下使用动作 $a$ 的期望累积折扣奖励值 $Q(s, a)$，训练以满足贝尔曼方程

$$Q(s_t, a_t) = r_{t+1} + \gamma(1 - d_t)Q(s_{t+1}, \pi(s_{t+1})),$$

在由演员 $\pi: S \to A$ 给出的策略下，该演员尝试估计 Q 最优策略 $\pi(s) = \arg\max_a Q(s, a)$；其中 $(s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1})$ 是一个经验元组（experience tuple），即从状态 $s_t$ 使用动作 $a_t$ 转移到 $s_{t+1}$ 并接收奖励 $r_{t+1}$ 和"完成"标志 $d_{t+1}$，从过去经验的缓冲区中选取。贝尔曼等式中的误差——评论家试图最小化的——称为时序差分误差（temporal difference error, TD error）。演员-评论家方法的许多变体存在，参见 [28]、[29]。

DDPG 训练是在线进行的。除了在真实车辆上建立此类缓冲区的基础设施（需要容忍缺失/错误的回合并可随时停止）之外，还可以通过从重放缓冲区中选择最"有信息量"的样本来加速强化学习。我们使用一种常用的方法——优先经验回放（prioritised experience replay）[30]：以与评论家产生的 TD 误差成比例的概率采样经验元组。用于此采样的权重在每个优化步骤中以最小开销更新；新样本被赋予无穷大的权重以确保所有样本至少被看到一次。

DDPG 是一种离策略学习算法（off-policy），意味着训练期间执行的动作来自与演员学习的最优策略不同的策略。这是为了获得最优策略窄分布之外的多样化状态-动作数据，从而增加鲁棒性。我们使用连续强化学习方法中实现这一目标的标准方法：探索策略通过将离散 Ornstein-Uhlenbeck 过程噪声 [31] 添加到最优策略中形成。因此在每一步，我们向最优动作添加噪声 $x_t$，由以下公式给出：

$$x_{t+1} = x_t + \theta(\mu - x_t) + \sigma\epsilon_t, \tag{4}$$

其中 $\theta$、$\mu$、$\sigma$ 为超参数，$\{\epsilon_t\}_t$ 为从正态分布 $\mathcal{N}(0, 1)$ 采样的独立同分布随机变量。

这些参数需要仔细调优，因为噪声的实用性和安全驾驶员的舒适度之间存在直接权衡。均值回复性强且方差较低的噪声更易预测，而方差较高的噪声能提供更好的状态-动作空间覆盖。

### C. 基于任务的训练架构

在真实世界环境中运行的全尺寸机器人车辆上部署强化学习算法，需要调整常见的训练流程，以应对驾驶员干预和影响训练的外部变量。

我们将算法的架构设计为一个简单的状态机，如图 2a 所示，由安全驾驶员控制不同的任务。我们定义了四种任务：训练（train）、测试（test）、撤销（undo）和完成（done）。这些任务的定义使系统既具有交互性又具有状态性，有利于按需执行回合而非预先固定的调度。

训练和测试任务允许我们在自主模式下与车辆交互，执行当前策略。两种任务的区别在于：训练任务中将噪声添加到模型输出且模型被优化，而测试任务直接运行模型输出的动作。在早期回合中，我们跳过优化以利于状态空间的探索。我们继续实验直到测试奖励不再增加。

每个回合执行直到系统检测到自动化丢失（即驾驶员干预）。在真实世界环境中，系统无法像在仿真或受限环境中的智能体那样在回合之间自动重置。我们需要人类驾驶员将车辆重置到有效的起始状态。在回合终止后，当安全驾驶员执行此重置时，模型正在被优化，从而最大限度地减少回合之间的时间。

撤销和完成任务体现了该架构中的关键差异。系统可能因多种有效原因（而非驾驶失败）终止一个回合：这些回合不能用于训练目的。撤销任务正是为此引入的，它允许我们撤销该回合并将模型恢复到运行该回合之前的状态。在我们实验中的一个常见例子是遇到其他希望使用该道路的驾驶员。完成任务允许我们在任何时刻优雅地退出实验，这在交互式流程中很有用，因为它不运行固定数量的回合。

> **图 2**：高效训练算法的工作流程和架构概要，基于安全驾驶员的反馈。(a) 展示了车载训练的基于任务的工作流程伪代码状态机。(b) 展示了策略执行架构，用于在模型训练或测试期间运行回合，包括有状态训练器、RL 模型、控制器和车辆之间的交互，分别以 10Hz 和 100Hz 运行。

---

## IV. 实验

我们用来展示车辆能力的主要任务是车道跟随；这与 [8] 中解决的任务相同，但在真实车辆和仿真中完成，且基于图像输入，无需知道车道位置。这是驾驶的核心任务，也是开创性的 ALVINN [19] 的基石。我们首先在第 IV-A 节的仿真中完成此任务，然后利用这些结果和适当的超参数知识在第 IV-B 节展示在真实车辆上的解决方案。

对于仿真和真实世界实验，我们使用一个小型卷积神经网络。我们的模型有四个卷积层，使用 $3 \times 3$ 卷积核、步幅为 2 和 16 个特征维度，在演员和评论家模型之间共享。然后我们将编码后的状态展平，为演员连接标量状态向量，评论家还额外连接动作。对于两个网络，我们在回归到输出之前都应用一个特征大小为 8 的全连接层。对于 VAE 实验，使用与编码器相同大小的解码器，用转置卷积替换步幅卷积以上采样特征。图形描述见图 1。

### A. 仿真

为了在图像输入的车道跟随场景中测试强化学习算法，我们使用虚幻引擎 4（Unreal Engine 4）开发了一个三维驾驶仿真器。它包含乡村道路的生成模型，支持多种天气条件和道路纹理，未来将支持更复杂的环境（游戏截图见图 3）。

> **图 3**：我们的车道跟随仿真器中每个回合随机生成的不同道路环境示例。我们使用程序化生成来随机变化每个回合的道路纹理、车道标记和道路拓扑。训练使用前向驾驶员视角图像作为输入。

仿真器对于调优强化学习参数至关重要，包括：学习率、每个训练回合后采取的梯度步数以及正确的终止程序——保守的终止策略能带来更好的策略。它证实了连续动作空间更优——离散动作导致抖动的策略——以及 DDPG 是一个合适的强化学习算法。

如第 III-A 节环境设置中所述，仿真器中给予的奖励对应于驶出车道前行驶的距离，新回合将汽车重置到车道中央。

我们发现可以在 10 个训练回合内可靠地从原始图像学会仿真中的车道跟随。此外，我们发现使用压缩状态表示（由变分自编码器提供）几乎没有优势。

我们发现以下超参数最为有效，并将其用于真实世界实验：未来折扣因子为 0.9，噪声半衰期为 250 个回合，噪声参数 $\theta$ 为 0.6、$\sigma$ 为 0.4，回合间 250 步优化、批量大小 64、梯度裁剪值 0.005。

### B. 真实世界驾驶

我们的真实世界驾驶实验在许多方面模仿了仿真中的实验。然而，在真实世界中执行此实验要困难得多。许多环境因素无法控制，且必须实施实时安全和控制系统。在这些实验中，我们使用了一段 250 米的道路。汽车从道路起点开始训练回合。当汽车偏离车道进入无法恢复的位置时，安全驾驶员接管车辆控制，结束该回合。然后将车辆返回车道中央开始下一个回合。我们使用在仿真中发现有效的相同超参数，噪声模型根据车辆本身的动力学特性进行调整，以使车辆行为与仿真中相似。

我们使用一辆改装的雷诺 Twizy 车辆进行实验，这是一辆双座电动车，如图 1 所示。该车重 500 公斤，最高时速 80 km/h，单次充电续航 100 公里。我们使用安装在车辆前部车顶中央的单个前向单目摄像头。我们使用改装的电动马达来驱动刹车和转向，并通过电子模拟油门位置来调节车轮扭矩。所有计算均在车载单台 NVIDIA Drive PX2 计算机上完成。

车辆的线控驱动自动化在安全驾驶员干预时自动解除，干预方式包括使用车辆控制器（刹车、油门或转向）、切换自动化模式或按下紧急停止按钮。当速度超过 10 km/h 或线控驱动自动化解除（表明安全驾驶员已干预）时，回合终止。安全驾驶员随后将汽车重置到道路中央，继续下一个回合。

> **图 4**：使用 VAE 与 DDPG 结合在训练中比从原始像素使用 DDPG 大大提高了数据效率，表明状态表示是在真实系统上应用强化学习的重要考虑因素。(a) 算法结果图：横轴为训练回合数，纵轴为自主行驶距离（米），展示了 DDPG 和 DDPG+VAE 的学习曲线，标注了随机策略和任务解决的基准线。(b) 用于实验的 250 米驾驶路线。

**表 I**：深度强化学习在 250 米道路上的自动驾驶车辆实验结果。

| 模型 | 训练回合数 | 训练距离 | 训练时间 | 测试：每次脱离行驶距离 (米) | 测试：脱离次数 |
|------|-----------|---------|---------|--------------------------|--------------|
| 随机策略 | - | - | - | 7.35 | 34 |
| 零策略（直行匀速） | - | - | - | 22.7 | 11 |
| 从像素的深度 RL | 35 | 298.8 m | 37 分钟 | 143.2 | 1 |
| 从 VAE 的深度 RL | 11 | 195.5 m | 15 分钟 | - | 0 |

我们报告了每个模型的最佳性能。我们观察到基线 RL 智能体可以从零开始学会车道跟随，而 VAE 变体效率更高，仅经过 11 个训练回合即可成功驾驶完整路线。

---

## V. 讨论

本工作展示了深度强化学习在全尺寸自动驾驶车辆上的首次应用。实验表明我们能够在不到三十分钟的训练中学会车道跟随——全部在车载计算机上完成。

为了调优超参数，我们构建了一个简单的仿真驾驶环境，在其中对强化学习算法进行了实验，使用 DDPG 作为经典算法最大化交通违规前的行驶距离。找到的参数成功迁移到真实世界，我们在私有道路上快速训练了一个策略来驾驶真实车辆，奖励信号仅包括速度和在控制驾驶员接管时终止。值得注意的是，这种奖励不需要环境的额外信息或地图。随着更多数据、车辆和更大的模型，该框架具有足够的通用性来扩展到更复杂的驾驶任务。

虽然可行，但如果要成为自动驾驶扩展的领先方法，该方法将需要将强化学习研究进展进行转化，以及在核心强化学习算法方面的工作。我们最后讨论对未来工作的思考。

在本工作中，我们提出了一个通用奖励函数，要求智能体最大化在安全驾驶员干预之前的行驶距离。虽然该奖励函数具有通用性，但它有一些局限性。它不考虑对给定导航目标的条件化。此外，它极其稀疏。随着智能体改进，干预将变得显著减少，导致训练信号减弱。可能需要进一步工作来设计更有效的奖励函数以学习超人驾驶智能体。这将涉及对许多安全 [32] 和伦理问题 [33] 的仔细考虑。

结果所建议的第二个发展领域是更好的状态表示。我们的实验表明，简单的变分自编码器大大提高了 DDPG 在驾驶真实车辆场景下的性能。超越像素空间自编码器的是大量解决图像有效压缩的计算机视觉研究：现有的语义分割、深度估计、自运动估计和像素流等领域的工作为驾驶场景中什么是重要的提供了出色的先验 [34]、[1]、[35]。这些研究需要与实际任务的强化学习方法相结合，包括无模型和基于模型的方法。

然而，仅靠无监督状态编码可能不够。为了以使策略仅用少量样本即可学习的方式压缩状态，需要关于状态（图像观测）哪些元素重要的信息。这些信息应来自奖励和终止信号。奖励和终止信息可以通过多种方式纳入编码中，但一个困难始终存在：功劳分配（credit allocation）。在特定时间步获得的奖励可能与过去许多时间步接收的观测有关。因此用于此应用的良好模型将包含时间组件。

两个可以大大提高强化学习应用于真实自动驾驶数据可用性的领域是半监督学习（semi-supervised learning）[35] 和域迁移（domain transfer）[36]。虽然只有一小部分驾驶数据可能具有与之关联的奖励和终止信号（因为获取这些成本很高），但图像嵌入——以及模型的其他方面——可以受益于从日常车辆的行车记录仪中捕获的驾驶数据。这些数据可用于预训练图像自编码器。在基于模型的 RL 系统中，这些也可用于近似状态转移函数，而半监督学习的进步可能允许我们在没有奖励/终止标签数据的情况下利用这些数据。另一方面，域迁移可能允许我们创建足够逼真的仿真，使来自这些仿真的数据可用于训练可直接迁移到真实汽车上的策略。

这里使用的算法有意选择了一种常见的经典方法，以展示强化学习可以多么容易地应用于驾驶。文献中已开发了许多改进方法，包括使用自然梯度（natural gradients）[37]。其他研究关注将观测更好地转换为状态，通常使用 RNN [38]、[39]，以及执行多步规划的方法，如 [40]。毫无疑问，这些方法可以提供更优越的性能。

基于模型的强化学习的新进展为自动驾驶研究提供了令人兴奋的替代方向，如 [22] 的工作在直接观察物理系统状态时展示了出色的模型性能。这可能为基于图像的领域带来显著好处。替代的基于模型方法包括 [41]，它学习仿真回合并在想象中学习。

我们希望本文能激励更多将强化学习研究应用于自动驾驶的工作，或许将其与模仿学习和控制理论等其他机器学习技术的元素相结合。这里的方法在半小时内解决了一个简单的驾驶任务——一天之内还能做什么呢？

---

## 参考文献

[1] V. Badrinarayanan, A. Kendall, and R. Cipolla, "Segnet: A deep convolutional encoder-decoder architecture for scene segmentation," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

[2] T. Ort, L. Paull, and D. Rus, "Autonomous vehicle navigation in rural environments without detailed prior maps," in International Conference on Robotics and Automation (ICRA), 2018.

[3] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. MIT Press, 1998.

[4] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. P. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, "Mastering the game of go with deep neural networks and tree search," Nature, vol. 529, no. 7587, pp. 484–489, 2016.

[5] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. P. Lillicrap, K. Simonyan, and D. Hassabis, "Mastering chess and shogi by self-play with a general reinforcement learning algorithm," CoRR, vol. abs/1712.01815, 2017.

[6] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, p. 529, 2015.

[7] S. Gu, E. Holly, T. Lillicrap, and S. Levine, "Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates," in Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017, pp. 3389–3396.

[8] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, "Continuous control with deep reinforcement learning," in International Conference on Learning Representations (ICLR), 2016.

[9] T. Kanade, C. Thorpe, and W. Whittaker, "Autonomous land vehicle project at cmu," in Proceedings of the 1986 ACM fourteenth annual conference on Computer science. ACM, 1986, pp. 71–80.

[10] R. S. Wallace, A. Stentz, C. E. Thorpe, H. P. Moravec, W. Whittaker, and T. Kanade, "First results in robot road-following." in IJCAI. Citeseer, 1985, pp. 1089–1095.

[11] M. Montemerlo, J. Becker, S. Bhat, H. Dahlkamp, D. Dolgov, S. Ettinger, D. Haehnel, T. Hilden, G. Hoffmann, B. Huhnke, et al., "Junior: The stanford entry in the urban challenge," Journal of field Robotics, vol. 25, no. 9, pp. 569–597, 2008.

[12] J. Levinson, J. Askeland, J. Becker, J. Dolson, D. Held, S. Kammel, J. Z. Kolter, D. Langer, O. Pink, V. Pratt, et al., "Towards fully autonomous driving: Systems and algorithms," in Intelligent Vehicles Symposium (IV), 2011 IEEE. IEEE, 2011, pp. 163–168.

[13] U. Franke, D. Gavrila, S. Gorzig, F. Lindner, F. Puetzold, and C. Wohler, "Autonomous driving goes downtown," IEEE Intelligent Systems and Their Applications, vol. 13, no. 6, pp. 40–48, 1998.

[14] S. Thrun, W. Burgard, and D. Fox, Probabilistic robotics. MIT press, 2005.

[15] C. Linegar, W. Churchill, and P. Newman, "Made to measure: Bespoke landmarks for 24-hour, all-weather localisation with a camera," in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), Stockholm, Sweden, May 2016.

[16] U. Muller, J. Ben, E. Cosatto, B. Flepp, and Y. L. Cun, "Off-road obstacle avoidance through end-to-end learning," in Advances in neural information processing systems, 2006, pp. 739–746.

[17] A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? the kitti vision benchmark suite," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[18] W. Maddern, G. Pascoe, C. Linegar, and P. Newman, "1 year, 1000 km: The oxford robotcar dataset," The International Journal of Robotics Research, vol. 36, no. 1, pp. 3–15, 2017.

[19] D. A. Pomerleau, "Alvinn: An autonomous land vehicle in a neural network," in Advances in neural information processing systems, 1989, pp. 305–313.

[20] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, et al., "End to end learning for self-driving cars," arXiv preprint arXiv:1604.07316, 2016.

[21] R. S. Sutton, A. G. Barto, et al., Reinforcement learning: An introduction. MIT press, 1998.

[22] M. Deisenroth and C. E. Rasmussen, "Pilco: A model-based and data-efficient approach to policy search," in Proceedings of the 28th International Conference on machine learning (ICML), 2011, pp. 465–472.

[23] G. Williams, N. Wagener, B. Goldfain, P. Drews, J. M. Rehg, B. Boots, and E. A. Theodorou, "Information theoretic mpc for model-based reinforcement learning," in Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017, pp. 1714–1721.

[24] M. Riedmiller, M. Montemerlo, and H. Dahlkamp, "Learning to drive a real car in 20 minutes," in Frontiers in the Convergence of Bioscience and Information Technologies, 2007. FBIT 2007. IEEE, 2007, pp. 645–650.

[25] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel, "Backpropagation applied to handwritten zip code recognition," Neural computation, vol. 1, no. 4, pp. 541–551, 1989.

[26] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," in The International Conference on Learning Representations (ICLR), 2014.

[27] D. J. Rezende, S. Mohamed, and D. Wierstra, "Stochastic backpropagation and approximate inference in deep generative models," in Proceedings of the 31st International Conference on machine learning (ICML), 2014.

[28] R. S. Sutton, D. A. McAllester, S. P. Singh, and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," in Advances in neural information processing systems, 2000, pp. 1057–1063.

[29] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, "Asynchronous methods for deep reinforcement learning," in International Conference on Learning Representations (ICLR), 2016.

[30] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, "Prioritized experience replay," in International Conference on Learning Representations (ICLR), 2015.

[31] G. E. Uhlenbeck and L. S. Ornstein, "On the theory of the brownian motion," Phys. Rev., vol. 36, pp. 823–841, Sep 1930.

[32] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, "Concrete problems in ai safety," arXiv preprint arXiv:1606.06565, 2016.

[33] J. J. Thomson, "The trolley problem," The Yale Law Journal, vol. 94, no. 6, pp. 1395–1415, 1985.

[34] A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[35] T. Zhou, M. Brown, N. Snavely, and D. G. Lowe, "Unsupervised learning of depth and ego-motion from video," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[36] K. Bousmalis, N. Silberman, D. Dohan, D. Erhan, and D. Krishnan, "Unsupervised pixel-level domain adaptation with generative adversarial networks," in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[37] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, "Trust region policy optimization," in International Conference on Machine Learning, 2015, pp. 1889–1897.

[38] M. Hausknecht and P. Stone, "Deep recurrent q-learning for partially observable mdps," CoRR, abs/1507.06527, 2015.

[39] M. Igl, L. Zintgraf, T. A. Le, F. Wood, and S. Whiteson, "Deep variational reinforcement learning for pomdps," in Proceedings of the 28th International Conference on machine learning (ICML), 2018.

[40] G. Farquhar, T. Rocktäschel, M. Igl, and S. Whiteson, "Treeqn and atreec: Differentiable tree planning for deep reinforcement learning," in International Conference on Learning Representations (ICLR), 2018.

[41] D. Ha and J. Schmidhuber, "World models," CoRR, vol. abs/1803.10122, 2018.
