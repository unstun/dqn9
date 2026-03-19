# 利用深度强化学习在 Gran Turismo Sport 中实现超人类表现

**作者：** Florian Fuchs, Yunlong Song, Elia Kaufmann, Davide Scaramuzza, Peter Dürr

**来源：** IEEE Robotics and Automation Letters, Preprint Version, Accepted February 2021

**DOI:** 10.1109/LRA.2021.3064284

---

## 摘要

自主赛车（autonomous car racing）是机器人学中的一项重大挑战。它为经典方法提出了根本性问题，例如在不确定动力学条件下规划最短时间轨迹（minimum-time trajectory），以及在车辆操控极限下进行控制。此外，圈速最小化这一稀疏目标的特性，以及从人类专家处收集训练数据的困难，也阻碍了研究者直接应用基于学习的方法来解决该问题。

在本工作中，我们提出了一种基于学习的自主赛车系统，利用高保真物理仿真、赛程进度代理奖励（course-progress proxy reward）和深度强化学习（deep reinforcement learning）。我们将系统部署在 Gran Turismo Sport（GTS）中——这是一款以逼真物理仿真著称的世界领先赛车模拟器，支持多种赛车和赛道，甚至被用于选拔真实赛车手。我们训练的策略在自主赛车性能上超越了内置AI此前的最佳表现，同时在超过50,000名人类玩家的数据集中优于最快的人类驾驶员。

**关键词：** 自主智能体（Autonomous Agents）、强化学习（Reinforcement Learning）

**补充视频：** https://youtu.be/Zeyv1bN9v4A

---

## I. 引言

自主赛车的目标是尽可能快地完成给定赛道，这要求智能体即使在车辆达到物理极限时也能生成快速且精确的动作。经典方法 [1]–[3] 通常将赛车问题解耦为轨迹规划（trajectory planning）和车辆控制（vehicle control）两个子问题。从轨迹规划角度看，问题在于如何为给定赛道计算最短时间轨迹；从车辆控制角度看，它涉及自动驾驶中最关键的场景——如何在极端操纵下安全控制车辆。

经典方法在自主赛车问题上已被广泛研究并展现出令人瞩目的成果。尤其是大量成功依赖于基于优化的轨迹生成与跟踪技术，得益于其处理不同约束和非线性动力学的能力。然而，这一研究方向存在若干局限，例如非线性优化所需的昂贵计算，以及目标函数缺乏灵活性（大多数优化要求二次代价形式）。总体而言，传统自动驾驶系统随着模型复杂度的增加而急剧膨胀，且未能充分利用真实世界驾驶产生的大量数据。

改进车辆模型以利用真实世界数据、处理更复杂的代价函数和一般非线性动力学的需求，促使研究者设计基于学习的系统。第一个趋势与动态模型辨识（dynamic model identification）有关。多层神经网络模型在与模型预测路径积分控制（Model Predictive Path Integral control, MPPI）[1], [4], [5] 或简单的前馈-反馈控制 [6] 结合时，已被证明有助于高速驾驶。然而，为了在快速控制回路中以滚动时域方式在线运行MPPI，优化需要高度并行的采样方案，严重依赖图形处理器（GPU）。

第二个趋势涉及使用神经网络的端到端（end-to-end）车辆控制，其中训练好的神经网络策略可以将高维原始观测直接映射到控制指令。因此，它具有无需显式状态估计和在线求解优化的优势，能够实现快速且自适应的控制性能。结合先进的动态模型辨识，神经网络控制器在许多真实世界机器人应用中取得了大量成就 [7]–[10]。

在本工作中，我们在 Gran Turismo Sport（GTS）中解决自主赛车问题。GTS 是一款以逼真建模各类赛车和赛道而闻名的世界领先赛车模拟器。GTS 是研究自主赛车系统的独特平台，因为它允许模拟大量真实环境，并在自主系统与经验丰富的人类驾驶员之间进行基准比较。这种比较能够为极端操纵下的车辆控制提供宝贵见解。

我们的方法使用无模型深度强化学习（model-free deep reinforcement learning）和赛程进度代理奖励来训练一个多层感知机（multilayer perceptron）策略用于车辆控制。关键在于将最短时间赛车问题恰当地转化为最大化赛程进度，并为高效策略训练选择有意义的低维状态表示。因此，我们展示了自主赛车任务中首个超人类控制性能。我们控制器的优势体现在两方面：1) 它不依赖高层轨迹规划与跟踪（人类驾驶员通常采用该方式）；2) 它生成的轨迹在质量上与最佳人类驾驶员选择的轨迹相似，同时超越了已知最快的人类圈速。我们的发现表明，强化学习和神经网络控制器值得在自主车辆操控极限方面进行更深入的研究。

---

## II. 相关工作

自主赛车领域的先前工作可分为三组：(i) 基于轨迹规划和跟踪的经典方法，(ii) 监督学习方法，(iii) 强化学习方法。

### 经典方法

经典自主赛车方法通过将问题分解为感知、轨迹规划和控制的子模块链来处理。模型预测控制（Model Predictive Control, MPC）[2], [11]–[15] 是在高速下控制车辆的一种有前途的方法。在 [16] 中，MPC 控制器与基于高斯过程的学习系统动力学相结合，用于自主赛车任务。虽然能够捕捉赛车的复杂动力学，但所得控制器需要在动力学保真度和计算时间之间进行权衡。类似地，MPPI [1], [4], [5] 是一种更灵活的方法，可以与复杂代价函数和神经网络车辆模型相结合。MPC 和 MPPI 在真实世界高速控制物理车辆方面都展现了出色的结果。尽管取得了成功，两种方法都存在局限，如代价函数设计缺乏灵活性或需要高度并行计算。此外，层次化方法也容易受到每个子模块故障的影响，其中每个模块的运行状态受到一系列近似和线性化的限制。

### 模仿学习

与规划轨迹并用控制器跟踪不同，基于模仿的方法（imitation learning）以监督方式直接学习从观测到控制动作的映射。学习这种映射需要标注数据，通常由人类专家演示或经典规划与控制流水线提供。例如，ALVINN（自主陆地车辆神经网络）[17] 是最早使用神经网络进行道路跟随的自动驾驶系统之一。类似地，[18] 训练了卷积神经网络（CNN）控制器用于车道和道路跟随。Pan 等人 [19] 使用模仿学习实现敏捷的越野自动驾驶。将人类或算法专家数据蒸馏到学习策略中是克服经典方法严格实时约束的有前途途径，但其性能在设计上以训练数据质量为上界。

### 强化学习

无模型强化学习（model-free reinforcement learning）直接基于采样轨迹优化参数化策略，因此不受前述方法所面临问题的困扰，如在线求解非线性优化的必要性或对标注训练数据的依赖。例如，若干研究 [20]–[24] 已展示了使用无模型深度强化学习进行端到端驾驶和赛车的成功案例。最近，一个高速自主漂移系统 [24] 在仿真中使用软演员-评论家（Soft Actor-Critic, SAC）算法 [25] 开发。离策略训练（off-policy training）在 [21], [22], [24] 中发挥了重要作用，其中通过将离线训练与经验回放缓冲区相结合，大幅降低了此前限制深度强化学习方法在高维领域广泛应用的高样本复杂度。尽管深度强化学习算法在真实和仿真自动驾驶中成功应用，但据我们所知，尚无工作在速度方面达到或超过人类专家驾驶员的表现。

---

## III. 方法

我们的主要目标是构建一个神经网络控制器，能够在不了解赛车动力学先验知识的情况下自主导航赛车，同时在给定 GTS 环境赛道上最小化行驶时间。为实现此目标，我们首先定义一个奖励函数来形式化赛车问题，以及一个将输入状态映射到动作的神经网络策略，然后使用 SAC [25] 算法通过最大化奖励函数来优化策略参数。系统概览见图 2。

> **图 2.** 系统概览：训练一个策略网络，将观测（包括一组测距仪测量值以及赛车的速度和加速度）直接映射到控制指令（赛车的转向角以及油门和制动级别）。使用分布式采样方案从4台 PlayStation 4 上收集样本（每台模拟20辆赛车），将采样轨迹存储在固定大小的先进先出（FIFO）回放缓冲区中。与此同时，使用软演员-评论家算法和从回放缓冲区均匀采样的数据并行优化策略参数。

### A. 最短时间问题与奖励函数

我们旨在找到一个策略，使给定赛车和赛道的总圈速最小。圈速本身是一个非常稀疏的奖励信号，信号的变化因此难以归因于智能体的特定动作。我们设计了一个基于当前赛程进度的代理奖励，可以在任意时间间隔内进行评估。为降低奖励信号的方差，我们对未来奖励使用指数折扣，折扣因子为 $\gamma$。当选择足够高的折扣因子时，最大化赛程进度可以很好地近似最小化圈速。新的代理奖励允许通过调整 $\gamma$ 来权衡一个容易归因但有偏的奖励与一个更接近整体圈速目标的奖励，从而根据未来奖励与动作和状态的时间距离改变其权重。进度奖励的构造见图 3 上部。

使用指数折扣的未来奖励会导致对短期奖励的偏好。在 GTS 环境中，这种偏好降低了强化学习智能体制动的动机，例如为防止碰撞而制动。为抵消这种短期偏好，我们引入第二个奖励项，根据赛车动能对撞墙进行惩罚，最终奖励函数为：

$$r_t = r^{prog}_t - \begin{cases} c_w \|v_t\|^2 & \text{if in contact with wall} \\ 0 & \text{otherwise} \end{cases} \tag{1}$$

其中 $c_w \geq 0$ 是一个超参数，控制撞墙惩罚与基于进度的奖励之间的权衡。$v_t = [v_x, v_y, v_z]$ 是表示车辆当前线速度的速度向量。类似的碰撞避免激励方法在 [26] 中被引入。引入动能的合理性在于撞墙时会发生与能量相关的加速度损失。没有这个额外的撞墙惩罚，学习到的策略不会制动，而是在急弯中沿赛道墙壁摩擦前进。当使用固定值的撞墙惩罚时，智能体要么对惩罚没有反应，要么采取完全制动并静止不动的策略以避免任何撞墙风险，具体取决于惩罚的强度。

> **图 3.** 上部：时刻 $t$ 的赛程进度 $cp_t$ 通过将赛车位置投影到赛道中心线上构造。然后将基于进度的奖励 $r^{prog}$ 定义为中心线进度的增量 $r^{prog}_t = cp_t - cp_{t-1}$。下部：馈入策略网络和价值网络的部分观测，包括边缘距离测量和速度分量。

### B. 策略网络

我们使用深度神经网络表示驾驶策略，采用原始论文 [25] 中提出的 SAC 网络架构。使用一个策略网络、两个 Q 函数网络和一个状态价值函数网络，每个网络包含 2 个隐藏层，各 256 个 ReLU 节点，总计 599,566 个可训练参数。

**输入特征：** 基于在 GTS 上进行的行为克隆（behavioral cloning）预研结果选择输入特征，该预研旨在通过对专家动作在已访问状态上的回归来学习类人驾驶。输入特征包括：

1. 线速度 $v_t \in \mathbb{R}^3$ 和线加速度 $\dot{v}_t \in \mathbb{R}^3$
2. 欧拉角 $\theta_t \in (-\pi, \pi]$——定义智能体水平面旋转的二维向量与投影点处中心线切向单位向量之间的夹角，这是策略网络检测赛车是否朝错误方向行驶的唯一直接方式
3. $M$ 个测距仪（rangefinder）的距离测量 $d_t \in \mathbb{R}^M$，最大量程 100 m，测量从车辆中心点到周围物体边缘点的距离。测距仪均匀分布在赛车前方 180° 视野内
4. 上一时刻的转向指令 $\delta_{t-1}$
5. 二值标志 $w_t = 1$ 表示撞墙
6. 近未来中心线的 $N$ 个采样曲率测量 $c_t \in \mathbb{R}^N$，曲率通过对 GTS 提供的中心线点拟合圆的逆半径进行插值表示

观测向量表示为 $s_t = [v_t, \dot{v}_t, \theta_t, d_t, \delta_{t-1}, w_t, c_t]$。为保证与人类驾驶员的公平比较，仅使用人类可以直接感知或从 GTS 模拟中推断的特征。

**网络输出：** 策略网络的输出 $a_t = [\delta_t, \omega_t]$ 直接编码转向角 $\delta_t \in [-\pi/6, \pi/6]$ rad 和油门-制动组合信号 $\omega_t \in [-1, 1]$，其中 $\omega_t = 1$ 表示全油门，$\omega_t = -1$ 表示全制动。将油门和制动合并为单一信号的动机来自对人类录像的分析——最快策略不涉及油门和制动的同时使用。

---

## IV. 实验

我们在三个赛车设定中评估我们的方法，涉及不同赛车和不同难度的赛道。将我们的方法与内置 AI 以及超过 50,000 名人类驾驶员进行比较。为确保与人类驾驶员的公平比较，我们约束策略仅生成物理方向盘可实现的动作。

### 赛车设定

为三个实验条件分别训练智能体，这些条件具有来自过去在线计时赛的人类数据。数据由 GTS 开发商 Polyphony Digital Inc. 提供，包含每位参赛者的个人最佳圈速和轨迹，范围从完全初学者到世界杯参赛者。比赛限制为固定设置，如赛车、赛道、轮胎和赛车辅助设置，从而在比较人类结果与我们的方法时确保相同条件。

> **图 4.** 三个设定使用的赛道和赛车。设定 A 和 B 使用的赛道布局具有弯道和直道的典型组合。两个设定使用的赛车不同：设定 A 的 "Audi TT Cup '16" 具有更高的最大速度、更强的轮胎抓地力和更快的加速度；设定 B 的 "Mazda Demio XD Turing '15" 相对较慢。设定 C 使用与设定 A 相同的赛车但赛道布局更具挑战性，由于长直道和非常紧的弯道组合而具有更大的速度范围。

### 公平竞争

人类玩家受限于游戏控制器的物理惯性，而我们的智能体原则上可以进行任意大的动作变化。为实现公平比较，我们估计了人类使用 Thrustmaster T300 方向盘和踏板在一帧内能实现的最大动作变化。基于此，将评估期间帧间最大变化限制为：转向角 0.03 rad，油门和制动范围各 80%。我们发现此限制对最终圈速无显著影响。

### 鲁棒性

为检测我们方法在更一般驾驶任务中的潜在不足，我们通过修改测试环境来评估智能体的鲁棒性。将设定 A 中训练的最快智能体在以下问题设定中部署，无需任何重新训练：1) 迁移到不同赛车（设定 B）；2) 迁移到不同赛道（设定 C）；3) 改变轮胎摩擦力；4) 向观测添加均匀噪声；5) 延迟智能体的推理。

---

## V. 结果

本节在第 IV 节介绍的三个参考设定上评估我们的方法，将圈速与 GTS 内置 AI 和最快人类驾驶员进行比较。此外，分析学习到的驾驶行为并与专业人类驾驶员的驾驶策略进行对比。

### A. 圈速比较

我们的方法在所有三个参考设定中均超越了最快人类圈速，克服了内置 AI 的局限（内置 AI 本身被大多数玩家超越）。

**表 I：三个赛车设定中我们的方法、人类在线参赛者和内置 GTS AI 的计时赛比较**

| 驾驶者 | 指标 | 设定 A | 设定 B | 设定 C |
|--------|------|--------|--------|--------|
| 我们的方法 | 圈速 [min] | 01:15.913 | 01:39.408 | 02:06.701 |
| 人类玩家 | 最快圈速 [min] | 01:16.062 | 01:39.445 | 02:07.319 |
| 人类玩家 | 中位圈速 [min] | 01:22.300 | 01:47.259 | 02:13.980 |
| 人类玩家 | 参赛人数 | 52,303 | 28,083 | 52,335 |
| 内置 GTS AI | 圈速 [min] | 01:26.899 | 01:52.075 | 02:14.252 |
| 内置 GTS AI | 慢于 x% 的人类驾驶员 | 82.6% | 80.4% | 53.9% |

在第一条赛道上，我们的方法分别比设定 A 和 B 中最快的人类圈速快 0.15 秒和 0.04 秒。我们认为与最佳人类驾驶员差距的不同源于两辆赛车的速度差异——设定 A 中较快的 "Audi TT Cup" 需要比设定 B 中相对较慢的 "Mazda Demio" 更敏捷的策略。虽然人类玩家可能在快节奏设定中挣扎，但我们的方法不受增加的需求影响。在设定 A 中连续 10 圈的标准差为 15 ms，而受邀参与研究的前 1% 驾驶员标准差为 480 ms。在设定 C 中，我们的方法比最快人类时间快 0.62 秒。更具挑战性的赛道布局和更快的赛车使设定 C 成为实验中最困难的组合。

> **图 5.** 设定 A 的训练进程：三个不同初始化的神经网络策略的学习曲线。每经过 2 个训练周期，让智能体驾驶连续 2 圈进行评估，取第二圈圈速作为性能指标。三个种子的方法均在 56 至 73 小时训练后学习到超越最快人类参考圈速的策略，平均对应 2,151 个训练周期和总共 946,453 公里行驶距离。值得注意的是，从模仿学习人类数据预训练的策略进行热启动并未改善最终性能——从头训练的策略在不到 1 小时后就超越了行为克隆策略。

### B. 学习到的驾驶行为

我们分析评估圈速最快的周期中模型的驾驶行为。

**外-内-外轨迹（Out-in-out trajectory）：** 我们的方法学会了利用赛道的全部宽度来最大化轨迹的弯道半径，驾驶所谓的外-内-外轨迹。这使智能体在失去抓地力之前能以更高速度行驶。图 6 展示了设定 A 中 3 个弯道的弯道驾驶比较，我们的方法学会了驾驶与最快人类相似的弯道轨迹，而无需任何人类演示或显式路径规划机制。

> **图 6.** 设定 A 中驾驶路径的俯视图：我们的方法和人类驾驶员的路径相似，而内置 GTS AI 与赛道墙壁保持安全裕度，导致更小的弯道半径和降低的速度。

**弯道预判：** 我们的方法还学会了足够早地检测弯道并评估其急程度，从而减速到能完成弯道而不冲出赛道墙壁的速度，同时又不过于保守。在设定 C 的发卡弯中，智能体在弯道起点约 100 米前就开始减速。

> **图 7.** 设定 A（上）和设定 C（下）中我们方法学习到的外-内-外驾驶行为和早期弯道预判。展示了进弯、切顶点和出弯的三个阶段。

**整体速度：** 图 8 上部显示了设定 A 中我们的方法、最快人类圈和内置 AI 之间的速度比较。我们的方法学习到的控制策略在速度上与最快人类参考圈密切匹配甚至有所提升。尽管设定 A 中路径和速度相似，我们的方法仍比最快人类圈速快 0.15 秒。虽然这看起来差距不大，但这一差距与真实和模拟赛车锦标赛中顶尖选手之间的差距相当。

> **图 8.** 上部：设定 A 中最快人类、我们的方法和内置 GTS AI 之间的速度比较。下部：油门/制动组合信号的比较。

> **图 9.** 左：设定 C 发卡弯段中我们的方法和最快人类驾驶的轨迹。虽然人类专家在入弯直道上驾驶更宽的轨迹，我们的方法在出弯直道上进行补偿，导致相似的弯道出口速度和弯道完成时间。右：我们的方法与最快人类在整条赛道上的速度差。正值表示我们的方法在该路段更快。

总而言之，这些结果表明我们的方法在不同赛车设定中（使用不同赛车、驾驶不同赛道，包括具有挑战性的发卡弯）学会了高速自主驾驶。在所有设定中，我们的方法均实现了快于所有人类参考驾驶员的圈速。此外，我们的方法学习到的轨迹在质量上与最佳人类玩家选择的轨迹相似，同时通过成功执行更晚的制动点在弯道中保持略高的平均速度。

### C. 鲁棒性

通过修改测试环境分析训练智能体的鲁棒性。

**1) 迁移到新赛车：** 智能体仍能遵循外-内-外轨迹并完成赛道。在直道和缓弯中，智能体能够适应动力学变化并执行可行轨迹。但在急弯中，智能体有时会冲出赛道，最可能的原因是训练用车比测试用车更容易控制（由于训练车较重且使用竞赛轮胎带来更高的轮胎-路面摩擦力）。

**2) 迁移到新赛道：** 智能体在直道和缓弯中能够无碰撞驾驶，但无法将行为外推到部分未见过的弯道形状。由于智能体在发卡弯中被卡住，无法完成赛道。

**3) 轮胎摩擦力变化：** 增大摩擦力时，智能体能够通过偏离预期轨迹时进行修正来适应，在 2 个弯道中高估了外力，导致与弯道内侧短暂接触，比基线慢 0.1 秒。降低摩擦力时，智能体能够修正轨迹以适应动力学变化，在除 3 个最急弯外的所有弯道中驾驶外-内-外路径。

**4) 观测噪声：** 图 10 下部显示了向智能体观测添加不同程度均匀噪声后的设定 A 圈速。噪声不超过 2% 时，智能体驾驶轨迹接近无噪声基线，仍领先最快人类。更大噪声导致动作信号抖动，产生速度损失。噪声不超过 9% 时，智能体仍能在赛道上无碰撞行驶。

**5) 推理延迟：** 图 10 上部显示了不同延迟程度下的圈速。延迟不超过 20 ms 时，性能接近基线，在每种情况下均超越最快人类参考圈。延迟 50 ms 时比基线慢 0.3 秒，但仍位列人类前 10 名。延迟 100 ms 时由于在最急弯前制动过晚导致撞墙，比基线慢 1.4 秒。延迟 150 ms 时无法驾驶有竞争力的赛车线路。

> **图 10.** 上部：推理延迟对圈速的影响。下部：观测噪声对圈速的影响。

在人机比较中，人类受约 0.2 秒相对较慢反应时间 [29] 的不利影响，而我们的智能体能在约 4 毫秒内做出反应（包括推理和与 PlayStation 的通信）。然而，我们认为这一劣势是人类系统的一部分，而非自主赛车挑战本身的一部分。

---

## VI. 结论

本文提出了首个在赛车模拟器 Gran Turismo Sport 的计时赛设定中实现超人类表现的自主赛车策略。我们方法的优势在于：不依赖人类干预、人类专家数据或显式路径规划；生成的轨迹在质量上与最佳人类玩家选择的轨迹相似，同时在所有三个参考设定中（包括两辆不同赛车和两条不同赛道）超越了已知最快的人类圈速。

这一超人类性能在评估和训练期间均使用了有限的计算能力。在标准台式电脑 CPU 上生成一个控制指令约需 0.35 ms，使用 4 台 PlayStation 4 游戏主机和一台台式电脑的训练在不到 73 小时内即实现超人类性能。

**局限性：** a) 限于单人计时赛，赛道上无其他赛车；b) 学习的控制策略仅适用于单一赛道/赛车组合。未来工作计划通过扩展观测空间以感知其他赛车并修改奖励来抑制不公平驾驶行为来解决 a)。为将方法扩展到更多赛道/赛车组合，建议使用更高数据效率的强化学习算法，如元强化学习（meta-RL），使智能体仅需少量新样本即可适应新情境。

---

## 致谢

非常感谢 Polyphony Digital Inc. 对本研究的支持。同时感谢索尼东京研发中心的 Kenta Kawamoto 和 Takuma Seno 的热心帮助和许多富有成效的讨论。

---

## 参考文献

[1] G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou, "Aggressive driving with model predictive path integral control," in 2016 IEEE International Conference on Robotics and Automation (ICRA), May 2016, pp. 1433–1440.

[2] A. Liniger, A. Domahidi, and M. Morari, "Optimization-based autonomous racing of 1: 43 scale rc cars," Optimal Control Applications and Methods, vol. 36, no. 5, pp. 628–647, 2015.

[3] E. Frazzoli, M. A. Dahleh, and E. Feron, "Real-time motion planning for agile autonomous vehicles," Journal of guidance, control, and dynamics, vol. 25, no. 1, pp. 116–129, 2002.

[4] G. Williams, N. Wagener, B. Goldfain, P. Drews, J. M. Rehg, B. Boots, and E. A. Theodorou, "Information theoretic mpc for model-based reinforcement learning," in 2017 IEEE International Conference on Robotics and Automation (ICRA), May 2017, pp. 1714–1721.

[5] G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou, "Information-theoretic model predictive control: Theory and applications to autonomous driving," IEEE Transactions on Robotics, vol. 34, no. 6, pp. 1603–1622, 2018.

[6] N. A. Spielberg, M. Brown, N. R. Kapania, J. C. Kegelman, and J. C. Gerdes, "Neural network vehicle models for high-performance automated driving," Science Robotics, vol. 4, no. 28, 2019.

[7] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, "Learning agile and dynamic motor skills for legged robots," Science Robotics, vol. 4, no. 26, p. eaau5872, 2019.

[8] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion over challenging terrain," Science Robotics, vol. 5, no. 47, 2020.

[9] J. Hwangbo, I. Sa, R. Siegwart, and M. Hutter, "Control of a quadrotor with reinforcement learning," IEEE Robotics and Automation Letters, vol. 2, no. 4, pp. 2096–2103, 2017.

[10] A. Nagabandi, K. Konolige, S. Levine, and V. Kumar, "Deep dynamics models for learning dexterous manipulation," in Conference on Robot Learning. PMLR, 2020, pp. 1101–1112.

[11] T. Novi, A. Liniger, R. Capitani, and C. Annicchiarico, "Real-time control for at-limit handling driving on a predefined path," Vehicle System Dynamics, vol. 58, no. 7, pp. 1007–1036, 2020.

[12] U. Rosolia and F. Borrelli, "Learning how to autonomously race a car: a predictive control approach," IEEE Transactions on Control Systems Technology, 2019.

[13] J. Kabzan, M. I. Valls, V. J. Reijgwart, H. F. Hendrikx, C. Ehmke, M. Prajapat, A. Bühler, N. Gosala, M. Gupta, R. Sivanesan et al., "Amz driverless: The full autonomous racing system," Journal of Field Robotics, vol. 37, no. 7, pp. 1267–1294, 2020.

[14] C. J. Ostafew, A. P. Schoellig, and T. D. Barfoot, "Robust constrained learning-based nmpc enabling reliable mobile robot path tracking," The International Journal of Robotics Research, vol. 35, no. 13, pp. 1547–1563, 2016.

[15] R. Verschueren, S. De Bruyne, M. Zanon, J. V. Frasch, and M. Diehl, "Towards time-optimal race car driving using nonlinear mpc in real-time," in 53rd IEEE conference on decision and control. IEEE, 2014, pp. 2505–2510.

[16] J. Kabzan, L. Hewing, A. Liniger, and M. N. Zeilinger, "Learning-based model predictive control for autonomous racing," IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3363–3370, 2019.

[17] D. A. Pomerleau, "Alvinn: An autonomous land vehicle in a neural network," in Advances in neural information processing systems, 1989, pp. 305–313.

[18] M. Bojarski, D. D. Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao, and K. Zieba, "End to end learning for self-driving cars," CoRR, vol. abs/1604.07316, 2016.

[19] Y. Pan, C.-A. Cheng, K. Saigol, K. Lee, X. Yan, E. Theodorou, and B. Boots, "Agile autonomous driving using end-to-end deep imitation learning," in Proceedings of Robotics: Science and Systems, Pittsburgh, Pennsylvania, June 2018.

[20] M. Jaritz, R. de Charette, M. Toromanoff, E. Perot, and F. Nashashibi, "End-to-end race driving with deep reinforcement learning," in 2018 IEEE International Conference on Robotics and Automation (ICRA), May 2018, pp. 2070–2075.

[21] M. Riedmiller, M. Montemerlo, and H. Dahlkamp, "Learning to drive a real car in 20 minutes," in 2007 Frontiers in the Convergence of Bioscience and Information Technologies, Oct 2007, pp. 645–650.

[22] A. Kendall, J. Hawke, D. Janz, P. Mazur, D. Reda, J. Allen, V. Lam, A. Bewley, and A. Shah, "Learning to drive in a day," in 2019 International Conference on Robotics and Automation (ICRA), May 2019, pp. 8248–8254.

[23] S. Grigorescu, B. Trasnea, T. Cocias, and G. Macesanu, "A survey of deep learning techniques for autonomous driving," Journal of Field Robotics, vol. 37, no. 3, pp. 362–386, 2020.

[24] P. Cai, X. Mei, L. Tai, Y. Sun, and M. Liu, "High-speed autonomous drifting with deep reinforcement learning," IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1247–1254, 2020.

[25] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in International Conference on Machine Learning, 2018, pp. 1856–1865.

[26] G. Kahn, A. Villaflor, V. Pong, P. Abbeel, and S. Levine, "Uncertainty-aware reinforcement learning for collision avoidance," CoRR, vol. abs/1702.01182, 2017.

[27] O. Vinyals, I. Babuschkin, W. M. Czarnecki, M. Mathieu, A. Dudzik, J. Chung, D. H. Choi, R. Powell, T. Ewalds, P. Georgiev et al., "Grandmaster level in starcraft ii using multi-agent reinforcement learning," Nature, vol. 575, no. 7782, pp. 350–354, 2019.

[28] V. Firoiu, T. Ju, and J. Tenenbaum, "At human speed: Deep reinforcement learning with action delay," CoRR, vol. abs/1810.07286, 2018.

[29] E. A. C. Thomas, "Information theory of choice-reaction times. by d. r. j. laming," British Journal of Mathematical and Statistical Psychology, vol. 22, no. 1, pp. 103–104, 1969.

[30] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.

[31] S. Ross, G. Gordon, and D. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," in Proceedings of the fourteenth international conference on artificial intelligence and statistics, 2011, pp. 627–635.

[32] C. Berner, G. Brockman, B. Chan, V. Cheung, P. Debiak, C. Dennison, D. Farhi, Q. Fischer, S. Hashme, C. Hesse et al., "Dota 2 with large scale deep reinforcement learning," arXiv preprint arXiv:1912.06680, 2019.

[33] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, "Mastering the game of go with deep neural networks and tree search," Nature, vol. 529, pp. 484–503, 2016.

[34] G. Barth-Maron, M. W. Hoffman, D. Budden, W. Dabney, D. Horgan, D. TB, A. Muldal, N. Heess, and T. Lillicrap, "Distributional policy gradients," in International Conference on Learning Representations, 2018.

[35] T. Hellstrom and O. Ringdahl, "Follow the past: a path-tracking algorithm for autonomous vehicles," International journal of vehicle autonomous systems, vol. 4, no. 2-4, pp. 216–224, 2006.

[36] P. Ritzer, C. Winter, and J. Brembeck, "Advanced path following control of an overactuated robotic vehicle," in 2015 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2015, pp. 1120–1125.

[37] J. Ni, J. Hu, and C. Xiang, "Robust path following control at driving/handling limits of an autonomous electric racecar," IEEE Transactions on Vehicular Technology, vol. 68, no. 6, pp. 5518–5526, 2019.

---

## 附录

### A. 先前方法

在项目早期阶段，我们使用了不同的学习方法，包括近端策略优化（Proximal Policy Optimization, PPO）[30] 和模仿学习。两种方法均未接近 SAC 所达到的性能。PPO 需要更多训练数据，且由于其状态无关的探索而遭受过早收敛问题。模仿学习方法受 DAgger 问题 [31] 困扰，其性能以训练数据为上界（训练数据来自人类玩家的前 1% 驾驶员轨迹）。

### B. GTS 访问

GTS 模拟器运行在 PlayStation 4 上，而我们的智能体运行在独立的台式电脑上。我们无法直接访问 GTS 模拟器，也不了解 GTS 建模的赛车动力学。相反，我们通过以太网连接的专用 API 与 GTS 交互。该 API 提供最多 20 辆模拟赛车的当前状态，并接受赛车控制指令（在收到下一条指令前持续有效）。与先前强化学习工作常并行运行数千个模拟不同 [9], [32], [33]，我们每台 PlayStation 只能运行一个模拟，本工作训练时仅使用 4 台 PlayStation。模拟器以实时运行，训练期间数据收集不能加速，也不能暂停以创建额外决策时间。模拟器状态更新频率为 60 Hz，但为减轻控制 20 辆赛车时 PlayStation 的负载，训练期间指令频率限制为 10 Hz。评估时切换为 1 辆赛车，智能体动作频率提升至 60 Hz。使用标准台式电脑（i7-8700 处理器，3.20 GHz）进行推理，GeForce GTX 1080 Ti 显卡进行反向传播。

### C. 网络训练

每个周期并行展开 4×20 条轨迹，每条 100 秒，使用 4 台 PlayStation 上各 20 辆赛车。为减少训练期间赛车之间的干扰，将智能体位置均匀分布在赛道上，初始速度 100 km/h——这可以加速训练，因为智能体能更快接近最大可行路段速度。

训练使用基于 TensorFlow 的 OpenAI SAC 实现。修改代码库以允许在展开期间异步学习，并将默认 1 步 TD 误差改为 5 步 TD 误差以稳定训练（同 [34]）。所有 4 个网络的组合训练每周期约 35 秒，总周期时间为 100 秒。由于 GTS 环境的实时特性，大规模超参数搜索不可行，因此大部分采用 OpenAI 实现的默认超参数，除下表所列参数外。

**表 II：SAC 算法和 GTS 模拟使用的超参数**

| 超参数 | 值 |
|--------|-----|
| 小批量大小 | 4,096 |
| 回放缓冲区大小 | $4 \times 10^6$ |
| 学习率 | $3 \times 10^{-4}$ |
| 每周期更新步数 | 5,120 |
| 奖励缩放 ($1/\alpha$) | 100 |
| "Mazda Demio" 的指数折扣 ($\gamma$) | 0.98 |
| "Audi TT Cup" 的指数折扣 ($\gamma$) | 0.982 |
| 撞墙惩罚缩放 ($c_w$) | $5 \times 10^{-4}$ |
| 测距仪数量 ($M$) | 13（每 15°） |
| 曲率测量数量 ($N$) | 10（每 0.2s） |

> 更密集的测距仪导致更慢的收敛但未在学习策略中显示额外改进。

### D. 曲率构造

为表示中心线的曲率，利用 GTS 模拟提供的中心线点。这些点的分布使得曲率较高的区域由更多点表示，从而仅需查看三个相邻点即可获得精确的曲率测量。通过定义每个中心线点的曲率为该点及其两个邻居确定的圆的逆半径来利用此特性。右转弯用负逆半径表示，左转弯用正逆半径表示。然后在中心线点之间插值，得到关于赛程进度的连续曲率表示。曲率采样从当前位置（以赛车当前速度估计）等间距分布在未来 1.0 至 2.8 秒。

> **图 11.** 曲率测量的构造：利用 GTS 提供的中心线点，通过拟合三点确定圆来计算逆半径，再插值得到连续曲率表示。

### E. 控制信号

动作向量 $a_t = [\delta_t, \omega_t]$ 由两个控制信号定义：转向角 $\delta_t \in [-\pi/6, \pi/6]$（rad）对应赛车前轮角度，以及油门-制动组合信号 $\omega_t \in [-1, 1]$，其中 $\omega > 0$ 表示油门，$\omega \leq 0$ 表示制动。油门或制动的幅度与绝对值 $\|\omega\|$ 成正比。例如 $\omega = 1$ 为 100% 油门，$\omega = -1$ 为 100% 制动，$\omega = 0$ 为无油门无制动。合并两个信号降低了任务复杂度。为将策略网络输出限制在有效范围内，在输出层应用 tanh 激活函数 [25]。

使用 GTS 提供的自动换挡，因为当前 API 不支持手动换挡。此选项同样对人类玩家可用，但有经验的玩家多使用手动换挡以获得更多控制。

### F. 领域专家的结果分析

为加深对所取得结果的理解，我们邀请 Gran Turismo 领域专家 TG（因匿名原因隐去姓名）在我们的参考设定中比赛，并将其表现与我们的方法进行比较。TG 在多个国内和国际比赛中取得过顶尖成绩，在我们的两个参考设定中取得了位于前 0.36% 和 0.23% 百分位的圈速。

当被问及对策略驾驶风格的看法时，TG 表示：

> "策略驾驶非常激进，但我认为这只有通过其精确的动作才有可能。理论上我也可以驾驶相同的轨迹，但在 1000 次中有 999 次，我尝试那条轨迹会导致撞墙，这会毁掉我整圈的时间，我不得不从头开始新的一圈。"

### G. 相比内置 AI 的改进

在三个参考设定中，内置 AI 被大多数人类玩家超越（见表 I）。内置 AI 使用基于规则的跟踪方法沿预定义轨迹行驶，类似于控制领域广泛研究的其他轨迹跟随方法 [35]–[37]。由于 GTS 动力学的非线性，轨迹的微小偏差会强烈改变新的最优轨迹，使得预计算然后跟踪最优轨迹在实践中不可行。内置 AI 通过使用保守的参考轨迹（包含与赛道边界的裕度）来解决此问题，当偏离轨迹时允许恢复。然而这也导致弯道半径更小，迫使 AI 减速以不失去抓地力，导致弯道出口速度比我们的方法和最快人类慢多达 50 km/h。通过学习灵活的驾驶策略而不施加任何显式限制，我们的方法克服了先前内置 AI 的缺陷，学习到了在速度和路径方面更接近人类专家驾驶员的驾驶策略。

### H. GTS 1.57 更新后的实验可复现性

从 2020 年 4 月 23 日发布的 GTS 1.57 版本起，"Audi TT Cup '16" 的模拟动力学发生了变化，使得本文设定 A 和 C 的可实现时间有所不同。在新版本下，与旧版本收集的设定 A 和 C 人类数据集的公平比较不再可能。

**表 III：GTS 1.57 版本下新设定 A 中我们的方法、人类在线参赛者和内置 GTS AI 的计时赛比较**

| 驾驶者 | 指标 | 更新后设定 A |
|--------|------|-------------|
| 我们的方法 | 圈速 [min] | 01:14.686 |
| 人类玩家 | 最快圈速 [min] | 01:14.775 |
| 人类玩家 | 中位圈速 [min] | 01:21.794 |
| 人类玩家 | 参赛人数 | 71,005 |
| 内置 GTS AI | 圈速 [min] | 01:26.356 |
| 内置 GTS AI | 慢于 x% 的人类驾驶员 | 82.5% |

建议未来与本工作比较时使用 kudosprime.com 提供的人类数据集作为设定 A 的替代。该比赛的模拟设置与本文设定 A 相同，但使用了 1.57+ 版本更新的赛车动力学和更新的轮胎（RM 替代 RH）。本文方法在该新设定上经过 5 天训练后达到 74.686 秒的圈速。
