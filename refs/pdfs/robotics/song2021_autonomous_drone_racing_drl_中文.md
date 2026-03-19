# 基于深度强化学习的自主无人机竞速

**作者：** Yunlong Song\*, Mats Steinweg\*, Elia Kaufmann, Davide Scaramuzza

> 本文已被 IEEE/RSJ 智能机器人与系统国际会议（IROS）2021（布拉格）录用。©IEEE

---

**图 1：** 一架四旋翼无人机（quadrotor）以高速穿越三条不同的竞速赛道：AlphaPilot（左）、Split-S（中）和 AirSim（右）。轨迹由经深度强化学习（deep reinforcement learning）训练的神经网络策略计算生成。

---

## 摘要

在许多机器人任务中，如自主无人机竞速（autonomous drone racing），目标是尽可能快地穿越一组航路点（waypoints）。该任务的一个关键挑战是规划时间最优轨迹（time-optimal trajectory），通常假设预先完全已知需要通过的航路点来求解。由此产生的解要么高度专用于单一赛道布局，要么由于对平台动力学的简化假设而次优。本文提出了一种新的四旋翼无人机近时间最优轨迹生成方法。利用深度强化学习和相对门观测（relative gate observations），我们的方法能够计算近时间最优轨迹并适应环境变化。相较于基于轨迹优化（trajectory optimization）的方法，我们的方法在非平凡赛道配置上表现出计算优势。所提方法在仿真和真实世界的一组竞速赛道上进行了评估，在物理四旋翼平台上达到了高达 60 km/h 的速度。

视频：https://youtu.be/Hebpmadjqn8

---

## I. 引言

近年来，自主四旋翼无人机的快速导航研究取得了巨大进展，不断推动飞行器执行更加激进的机动动作 [1]–[3]。为了进一步推动该领域发展，多项竞赛相继举办，包括近年 IROS 和 NeurIPS 会议上的自主无人机竞速系列赛 [4]–[7] 以及 AlphaPilot 挑战赛 [8], [9]，其目标是开发最终能够超越人类专业飞手的自主系统。

将四旋翼无人机推向物理极限带来了具有挑战性的研究问题，包括规划穿越一系列门的时间最优轨迹。直觉上，这种时间最优轨迹探索了平台性能包络的边界。必须确保规划的轨迹满足执行器（actuators）施加的所有约束，否则控制权的减弱将导致灾难性的坠毁。此外，由于四旋翼无人机的线性和角度动力学是耦合的，计算时间最优轨迹需要在最大化线性加速度和角加速度之间进行权衡。

为了生成时间最优的四旋翼轨迹，先前的工作要么依赖于通过时间离散化轨迹的数值优化 [1]，要么通过多项式表达状态演化 [10]，甚至将四旋翼近似为质点（point mass）[9]。虽然基于优化的方法能够找到不断将平台推向极限的轨迹，但其计算时间极长，达到数小时级别。相比之下，使用多项式表示 [10] 或质点近似 [9] 的方法计算速度更快，但无法考虑平台真实的执行器极限，导致次优或动力学上不可行的解。

鉴于深度强化学习（deep RL）在机器人领域的最新成功 [11]–[13]，其有潜力解决四旋翼无人机的时间最优轨迹规划问题。特别是，无模型策略搜索（model-free policy search）通过直接与环境交互来学习参数化策略，非常适合难以建模的复杂问题。此外，策略搜索允许优化高容量的神经网络策略，可以接受不同形式的状态表示作为输入，并在算法设计中提供灵活性。例如，学习到的策略可以将机器人的原始感知观测直接映射到电机指令。

由此自然产生了若干研究问题，包括如何利用深度强化学习解决时间最优轨迹规划问题，以及其性能与基于模型方法的比较。

### 贡献

我们的主要贡献是一个基于强化学习的系统，能够计算极其激进的轨迹，这些轨迹接近于最先进轨迹优化方法 [1] 提供的时间最优解。据我们所知，这是第一个能够适应环境变化的学习型时间最优四旋翼轨迹规划方法。我们进行了大量实验，包括在确定性赛道、具有大不确定性的赛道以及随机生成的赛道上的规划。

我们的实证结果表明：1) 深度强化学习能够有效解决轨迹规划问题，但牺牲了轨迹优化能够提供的性能保证；2) 学习到的神经网络策略能够处理大规模赛道变化并在线重新规划轨迹，这对于基于轨迹优化的方法而言极具挑战性且计算代价高昂；3) 我们的神经网络策略展示的泛化能力允许适应不同的竞速环境，表明我们的策略可以部署到动态和非结构化环境中用于在线时间最优轨迹生成。

我们方法的关键要素包含三方面：1) 使用适当的状态表示，即相对门观测作为策略输入；2) 利用高度并行化的采样方案；3) 最大化三维路径进度奖励（path progress reward），这是最小化圈速时间的代理指标。

---

## II. 相关工作

现有的时间最优轨迹规划工作可分为基于采样（sampling-based）和基于优化（optimization-based）的方法。基于采样的算法，如 RRT\* [14]，在无限采样的极限下是可证明最优的。由于此类算法依赖于图的构建，通常与提供状态到状态轨迹闭合形式解的方法相结合。这一方向存在诸多挑战；例如，在考虑执行器约束时，两个完整四旋翼状态之间的闭合形式解难以解析推导。因此，先前的工作 [9], [15]–[17] 倾向于使用质点近似或多项式表示来描述状态演化，前者可能导致不可行的解，后者则如 [1] 所示导致固有的次优解。

关于四旋翼航路点的时间最优规划 [1] 使用离散时间状态空间表示来求解轨迹，并将系统动力学和推力限制作为约束。基于优化方法的一个主要优势，如 [1], [10], [18]，是它们能够融入非线性动力学并处理多种状态和输入约束。对于穿越多个航路点的情况，通过时间的分配先验未知，使得问题表述显著复杂化。求解复杂约束优化问题的需求使得大多数轨迹优化方法仅适用于可以离线计算时间最优轨迹的应用场景。

深度强化学习已被应用于自主导航任务，如高速竞速 [19], [20] 和自主车辆的运动规划 [21], [22]。这些方法的主要关注点是解决轮式机器人的导航问题，其运动受限于二维平面。然而，为四旋翼无人机找到时间最优轨迹更具挑战性，因为高维状态和输入空间使得搜索空间显著增大。先前将强化学习应用于四旋翼控制的工作展示了成功的航路点跟踪 [12] 和自主着陆 [23]，但这些工作都未能将平台推向其物理极限。我们的工作利用深度强化学习解决了以最短时间驾驶四旋翼穿越一组航路点的问题。我们的策略不对赛道布局施加强先验（strong priors），体现了其对动态环境的适用性。

---

## III. 方法

### A. 四旋翼动力学

我们将四旋翼无人机建模为质量为 $m$、对角惯性矩阵（diagonal moment of inertia matrix）为 $J$ 的 6 自由度刚体。系统动力学可以写为：

$$\dot{p}_{WB} = v_{WB}$$

$$\dot{q}_{WB} = \frac{1}{2}\Lambda(\omega_B) \cdot q_{WB}$$

$$\dot{v}_{WB} = q_{WB} \odot c + g$$

$$\dot{\omega}_B = J^{-1}(\eta - \omega_B \times J\omega_B)$$

其中 $p_{WB} = [p_x, p_y, p_z]^T$ 和 $v_{WB} = [v_x, v_y, v_z]^T$ 分别是四旋翼在世界坐标系 $W$ 中的位置和速度向量。使用单位四元数 $q_{WB} = [q_w, q_x, q_y, q_z]^T$ 表示四旋翼的姿态，$\omega_B = [\omega_x, \omega_y, \omega_z]^T$ 表示机体坐标系 $B$ 中的角速度。其中 $g = [0, 0, -g_z]^T$，$g_z = 9.81\ \text{m s}^{-2}$ 为重力向量，$J$ 为惯性矩阵，$\eta$ 为三维力矩，$\Lambda(\omega_B)$ 为反对称矩阵。$c = [0, 0, c]^T$ 为质量归一化推力向量。

单旋翼推力 $[f_1, f_2, f_3, f_4]$ 到质量归一化推力 $c$ 和机体力矩 $\eta$ 的转换公式为：

$$c = \frac{1}{m}\sum_{i=1}^{4}f_i$$

$$\eta = \begin{bmatrix} l/\sqrt{2}(f_1 - f_2 - f_3 + f_4) \\ l/\sqrt{2}(-f_1 - f_2 + f_3 + f_4) \\ \kappa(f_1 - f_2 + f_3 - f_4) \end{bmatrix}$$

其中 $m$ 为四旋翼质量，$\kappa$ 为旋翼扭矩常数，$l$ 为臂长。四旋翼的完整状态定义为 $x = [p_{WB}, q_{WB}, v_{WB}, \omega_B]$，控制输入为 $u = [f_1, f_2, f_3, f_4]$。我们使用四阶 Runge-Kutta 方法对动力学方程进行数值积分，记为 $f_{RK4}(x, u, dt)$，其中 $dt$ 为积分时间步长。

### B. 任务形式化

我们在强化学习框架中形式化时间最优轨迹规划问题。为此，我们使用无限时域马尔可夫决策过程（infinite-horizon Markov Decision Process, MDP）对任务建模，定义为元组 $(S, A, P, r, \rho_0, \gamma)$ [24]。强化学习智能体从初始状态分布 $\rho_0(s)$ 中采样得到起始状态 $s_t \in S$。在每个时间步 $t$，执行从随机策略 $\pi(a_t|s_t)$ 中采样的动作 $a_t \in A$，智能体以状态转移概率 $P^{a_t}_{s_t s_{t+1}} = \Pr(s_{t+1}|s_t, a_t)$ 转移到下一状态 $s_{t+1}$，并获得即时奖励 $r(t) \in \mathbb{R}$。状态和动作的具体表示在第 III-B.2 节讨论。

深度强化学习的目标是优化神经网络策略 $\pi_\theta$ 的参数 $\theta$，使训练后的策略最大化无限时域上的期望折扣奖励。目标的离散时间形式为：

$$\pi^*_\theta = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty}\gamma^t r(t)\right] \tag{1}$$

其中 $\gamma \in [0, 1)$ 为折扣因子（discount factor），用于权衡长期奖励与短期奖励。最优轨迹 $\tau^*$ 通过展开训练后的策略获得。

#### 1) 奖励函数

直觉上，在赛道上花费的总时间直接体现了任务的主要目标。然而，该信号只能在成功完成一圈后计算，引入的稀疏性（sparsity）大大增加了将功劳分配给个别动作的难度。一种常用的解决方法是使用一个代理奖励（proxy reward），在每个时间步向智能体提供反馈，同时近似地逼近真正的性能目标。在赛车竞速中，基于沿赛道中心线投影进度的奖励塑形（reward shaping）技术已被证明能很好地近似真实竞速奖励 [19], [25]。

我们将基于投影的路径进度奖励概念扩展到无人机竞速任务，使用连接相邻门中心的直线段作为赛道中心线的表示。因此，不需要额外计算参考路径。竞速赛道完全由放置在三维空间中的一组门定义。在任意时刻，四旋翼可以根据下一个需要通过的门关联到特定的线段。为计算路径进度，我们将四旋翼的位置投影到当前线段上。给定四旋翼的当前位置 $p_c(t)$ 和前一位置 $p_c(t-1)$，定义进度奖励 $r_p(t)$ 如下：

$$r_p(t) = s(p_c(t)) - s(p_c(t-1)) \tag{2}$$

其中 $s(p) = (p - g_1) \cdot (g_2 - g_1) / \|g_2 - g_1\|$ 定义了沿连接前一门中心 $g_1$ 与下一门中心 $g_2$ 的路径段的进度。

**图 2：** 进度奖励的示意图。展示了四旋翼在不同时间步 $p_c(t-2), p_c(t-1), p_c(t), p_c(t+1)$ 的位置沿连接门 1 和门 2 的线段的投影进度。

此外，我们定义了一个安全奖励（safety reward），通过惩罚较小的安全裕度来激励飞行器穿过门中心。需要注意的是，安全奖励是一个可选的奖励组件，旨在减少具有大赛道变化的训练设置中的坠毁风险。安全奖励定义如下：

$$r_s(t) = -f^2 \cdot \left(1 - \exp\left(-0.5 \cdot \frac{d_n^2}{v}\right)\right) \tag{3}$$

其中 $f = \max[1 - (d_p / d_{max}), 0.0]$，$v = \max[(1-f) \cdot (w_g / 6), 0.05]$。这里 $d_p$ 和 $d_n$ 分别表示四旋翼到门法线（gate normal）的距离和到门平面（gate plane）的距离。到门法线的距离用矩形门的边长 $w_g$ 进行归一化，而 $d_{max}$ 指定激活安全奖励的到门中心距离阈值。

**图 3：** 安全奖励的示意图。展示了 $d_n$（到门法线的距离）、$d_p$（到门平面的距离）和 $w_g$（门边长）的几何关系。

每个时间步 $t$ 的最终奖励定义为：

$$r(t) = r_p(t) + a \cdot r_s(t) - b \cdot \|\omega_t\|^2 + \begin{cases} r_T & \text{若门碰撞} \\ 0 & \text{否则} \end{cases} \tag{4}$$

其中 $r_T$ 为惩罚门碰撞的终止奖励（terminal reward）：

$$r_T = -\min\left[\frac{d_g}{w_g}^2, 20.0\right] \tag{5}$$

$d_g$ 和 $w_g$ 分别为坠毁位置到门中心的欧氏距离和矩形门的边长。此外，在初始训练阶段添加了以 $b \in \mathbb{R}$ 为权重的角速度 $\omega$ 二次惩罚项。$a \in \mathbb{R}$ 是一个超参数，用于在进度最大化和风险最小化之间进行权衡。

#### 2) 观测空间与动作空间

观测空间（observation space）由两个主要部分组成：一个与四旋翼状态 $s_t^{quad}$ 相关，另一个捕获赛道信息 $s_t^{track}$。四旋翼状态的观测向量定义为：

$$s_t^{quad} = [v_{WB,t}, \dot{v}_{WB,t}, R_{WB,t}, \omega_{B,t}] \in \mathbb{R}^{18}$$

分别对应四旋翼的线速度、线加速度、旋转矩阵和角速度。为避免姿态表示中的歧义性和不连续性，我们使用旋转矩阵来描述四旋翼姿态，这与先前使用强化学习进行四旋翼控制的工作一致 [12]。

赛道观测向量定义为 $s_t^{track} = [o_1, \alpha_1, \cdots, o_i, \alpha_i, \cdots]$，$i \in [1, \cdots, N]$，其中 $o_i \in \mathbb{R}^3$ 表示第 $i$ 个门的观测，$N \in \mathbb{Z}^+$ 为未来门的总数。使用球坐标 $o_i = (p_r, p_\theta, p_\phi)_i$ 表示下一个门的观测，该表示提供了到门距离和飞行方向之间的清晰分离。对于下一个需要通过的门，使用以机体为中心的参考坐标系，而所有其他门的观测递归地在前一个门的坐标系中表达。$\alpha_i \in \mathbb{R}$ 指定门法线与从四旋翼指向第 $i$ 个门中心的向量之间的角度。赛道观测的通用形式允许在观测中纳入可变数量的未来门。使用 z-score 归一化进行输入归一化，观测统计量在 1000 条随机生成的三维竞速赛道上的确定性策略展开中计算。

**图 4：** 本文使用的观测向量和网络架构示意图。

智能体被训练为直接将观测映射到各旋翼推力指令，动作定义为 $a_t = [f_1, f_2, f_3, f_4]$。使用各旋翼指令可实现最大的灵活性和激进飞行机动。在策略网络的最后一层使用 Tanh 激活函数，以将控制指令保持在固定范围内。

### C. 策略训练

我们使用近端策略优化（Proximal Policy Optimization, PPO）算法 [26] 训练智能体，这是一种一阶策略梯度方法，因其良好的基准性能和实现简洁性而特别流行。由于搜索空间的高维性和所需机动的复杂性，该任务对 PPO 提出了挑战。以下三个关键要素使我们能够在具有任意赛道长度和门数的复杂三维竞速赛道上实现稳定和快速的训练性能：

#### 1) 并行采样方案

在仿真中训练智能体允许在多达 100 个并行环境中执行策略展开，显著加速了数据收集过程。此外，并行化可以用来增加所收集环境交互的多样性。在长赛道上训练时，可以设计初始化策略，使展开轨迹覆盖整个赛道，而不是局限于赛道的某些部分。类似地，在应用强赛道随机化的设置中，可以在使用不同环境的多种赛道上执行展开，有效避免对特定赛道配置的过拟合。

#### 2) 分布式初始化策略

在学习过程的早期阶段，大多数展开因门碰撞而终止。因此，如果每架四旋翼都在起始位置初始化，数据收集将被限制在状态空间的一小部分，需要大量更新步骤直到策略能够探索完整赛道。为应对这种情况，我们采用分布式初始化策略（distributed initialization strategy），确保对状态空间相关区域的均匀探索。我们在所有路径段中心附近以悬停状态随机初始化四旋翼，使其立即暴露于所有门观测。一旦策略学会可靠地通过门，我们从先前的轨迹中采样初始状态。这样，我们保留了在整个赛道上初始化的优势，同时避免了在赛道高速区域从悬停位置起始的负面影响。

#### 3) 随机赛道课程

为单一赛道训练竞速策略本身已构成一个具有挑战性的问题。然而，鉴于赛道的静态性质，适当的初始化策略可以在不额外引导训练过程的情况下解决该问题。当超越使用确定性竞速赛道进行训练的设置时，任务复杂度急剧增加。从本质上讲，通过随机采样竞速赛道来训练策略，使智能体在每个回合都面临一个新任务。

为此，我们首先通过串联一组随机生成的门原语（gate primitives）来设计赛道生成器。赛道生成器的数学定义为：

$$T = [G_1, \cdots, G_{j+1}], \quad j \in [1, +\infty) \tag{6}$$

其中 $G_{j+1} = f(G_j, \Delta p, \Delta R)$ 为门原语，通过两个相邻门之间的相对位置 $\Delta p$ 和方向 $\Delta R$ 参数化。通过调整相对位姿的范围和门的总数，可以生成任意复杂度和长度的竞速赛道。

为设计一个在随机生成赛道的同时仍聚焦于最小化圈速时间的训练过程，我们提出根据智能体的表现自动调整采样赛道的复杂度。具体来说，训练开始时采样仅与直线有微小偏差的赛道。策略很快学会可靠地通过门，从而降低导致门碰撞的展开比例。我们使用在所有并行环境中计算的碰撞比率来调整采样赛道的复杂度，允许更多样化的相对门位姿。这样，数据收集被约束在特定类别的赛道上，直到达到指定的性能阈值。赛道复杂度的调整可以解释为一种自动课程学习（automatic curriculum），产生一个根据智能体能力量身定制的学习过程。

---

## IV. 实验

我们设计实验以回答以下研究问题：(i) 我们的学习方法在确定性赛道布局上取得的圈速时间与优化算法相比如何？(ii) 我们的方法能多好地处理赛道布局的变化？(iii) 是否可以训练一个能在完全未知赛道上成功竞速的策略？(iv) 我们策略生成的轨迹能否在物理四旋翼上执行？

### A. 实验设置

我们使用 Flightmare 仿真器 [27]，实现了一个向量化的 OpenAI Gym 风格竞速环境，可以并行仿真数百架四旋翼和竞速赛道。这种并行化实现使我们在训练过程中每秒可收集多达 25000 次环境交互。我们的 PPO 实现基于 [28]。

为了基准测试我们方法的性能，使用了三条不同的竞速赛道，包括 2019 年 AlphaPilot 挑战赛 [8] 的赛道、2019 年 NeurIPS AirSim Game of Drones 挑战赛 [7] 的赛道，以及为真实世界实验设计的赛道（Split-S）。这些赛道（图 5）为自主竞速无人机提出了多样化的挑战：AlphaPilot 赛道具有长直线段和发卡弯；AirSim 赛道需要执行快速且非常大的高度变化；Split-S 赛道引入了两个垂直叠放的门。

为测试方法的鲁棒性，我们在 AlphaPilot 赛道上对所有门的位置和偏航角使用随机位移。此外，我们使用第 III-C.3 节介绍的赛道生成器在完全随机的赛道上训练和评估我们的方法，以研究方法的可扩展性和泛化能力。最后，通过消融研究验证设计选择，并在物理平台上执行生成的轨迹。所有实验使用对角惯性矩阵 $J = [0.003, 0.003, 0.005]\ \text{kg m}^2$、扭矩常数 $\kappa = 0.01$ 和臂长 0.17 m 的四旋翼配置。对于 AlphaPilot 和 AirSim 赛道，使用 6.4 的推重比（thrust-to-weight ratio），而 Split-S 赛道使用 3.3 的推重比（用于真实世界部署）。除非另有说明，所有实验在观测中使用两个未来门。第 IV-E.2 节给出了关于门数量的消融研究。

**图 5：** 用于基线比较的竞速赛道及我们方法生成的轨迹。从左到右：AlphaPilot 赛道、Split-S 赛道、AirSim 赛道。颜色对应四旋翼的速度（范围 0–35 m/s）。

### B. 确定性赛道上的基线比较

我们首先在三条确定性赛道（第 IV-A 节）上将策略性能与两种轨迹规划算法进行基准对比：多项式最小急动度轨迹生成（polynomial minimum-snap trajectory generation）[29] 和基于互补进度约束（complementary progress constraints, CPC）的优化时间最优轨迹生成 [1]。多项式轨迹生成，特别是最小急动度轨迹生成的具体实例，由于其隐式平滑性而被广泛用于四旋翼飞行。虽然平滑性通常是一个非常有用的特性，但它禁止在每个时间点充分利用执行器潜力，使该方法对于竞速而言次优。相比之下，基于优化的方法（CPC）为给定赛道和飞行器模型提供了可能实现时间的渐近下界。我们不与基于采样的方法进行比较，因为它们对四旋翼动力学做了简化假设。据我们所知，不存在考虑四旋翼完整状态空间的基于采样的方法。

**表 I：圈速时间（秒）**

| 方法 | AlphaPilot (s) | Split-S (s) | AirSim (s) |
|------|---------------|-------------|------------|
| 多项式 [29] | 12.23 | 15.13 | 23.82 |
| 优化 [1] | 8.06 | 6.18 | 11.40 |
| 本文方法 | 8.14 | 6.50 | 11.82 |

结果汇总于表 I。我们的方法在 CPC 所示的理论极限的 5.2% 以内。使用多项式轨迹生成计算的基线明显更慢。图 5 展示了三条评估赛道及我们方法生成的相应轨迹。

### C. 处理赛道变化

虽然确定性赛道布局上的结果允许与理论极限进行比较，但在赛道布局事先不完全已知的更现实设置中，这样的圈速时间不可能获得。

我们设计了一个实验来研究方法对部分未知赛道布局的鲁棒性，通过在前一实验的 AlphaPilot 赛道上指定随机门位移。具体来说，使用有界位移 $[\Delta x, \Delta y, \Delta z, \Delta\alpha]$ 分别作用于每个门中心的位置和门在水平面上的朝向。图 6 展示了应用于原始赛道布局的显著位移范围。

我们通过从各位移边界的均匀分布中采样不同的门布局从头训练策略。每个训练迭代创建总共 100 个不同赛道配置的并行竞速环境。最终模型根据在 AlphaPilot 赛道标准版本上的最快圈速选择。定义两个指标来评估训练策略的性能：平均圈速时间和碰撞比率。在三个难度级别（由赛道随机性定义）上计算两个指标，每个级别包含 1000 个未见过的随机生成赛道配置。

**表 II：处理赛道不确定性的评估结果。门随机化级别对应于最大门位移的百分比。**

| 门随机化级别 | 平均时间 (s) | 碰撞次数 |
|-------------|-------------|---------|
| 0.00 | 8.32 | 0 |
| 0.50 | 8.43 | 0 |
| 0.75 | 8.55 | 2 |
| 1.00 | 8.73 | 25 |

结果如表 II 所示。在最困难的测试集（具有最大门位移）上，训练策略达到了 97.5% 的成功率。标准赛道上的圈速性能与无位移训练的策略（表 I）相比下降了 2.21%。为说明任务的复杂性，图 6 展示了在使用最大门位移的随机采样赛道上的 100 条成功轨迹以及标准轨迹。

**图 6：** 左：AlphaPilot 赛道的赛道变化。门中心可移动到各自立方体内的任意位置，蓝色锥体表示可能的朝向。右：AlphaPilot 赛道不同配置下的轨迹。颜色方案用于区分不同轨迹，不反映任何与轨迹相关的指标。标准轨迹以黄色显示。

### D. 迈向通用竞速策略

我们通过设计实验来研究方法的可扩展性和泛化能力，训练一个能够为完全未见赛道生成时间最优轨迹的策略。

为此，在每次展开中使用完全随机的赛道布局训练三个不同的策略。三个策略在训练策略和使用的奖励上有所不同：我们消融了自动赛道适应和安全奖励的使用。训练过程中，在 100 条未见的随机采样赛道的验证集上持续评估当前策略的性能。对于每种配置，选择在验证集上碰撞比率最低的策略作为最终模型。最终评估在 1000 条全复杂度随机采样赛道的测试集上进行，基于门碰撞次数评估性能。

**表 III：在随机生成竞速赛道上的性能。报告 1000 条测试赛道上的碰撞次数。**

| 安全奖励 | 赛道适应 | 碰撞次数 |
|---------|---------|---------|
| ✗ | ✗ | 71 |
| ✗ | ✓ | 47 |
| ✓ | ✓ | 26 |

实验结果如表 III 所示。在不使用安全奖励和自动赛道适应的情况下，策略在碰撞次数方面表现最差。添加自动赛道适应通过在训练开始时降低任务复杂度并在采样随机赛道时考虑当前学习进度，显著提高了性能。通过添加安全奖励，我们进一步降低了碰撞比率，说明了在随机赛道上训练时最大化安全裕度的积极效果。我们的策略在 1000 条随机生成的赛道上达到 97.4% 的成功率，展现了强大的泛化能力和可扩展性。

**图 7：** 在随机生成赛道上的策略展开（仅在 xy 平面可视化）。轨迹长度从 110 m 到 150 m，高度变化达 17 m。

### E. 计算时间比较和消融研究

#### 1) 计算时间比较

我们在受控实验中将强化学习方法的训练时间与轨迹优化方法的计算时间进行比较。具体来说，比较第 IV-B 节中三条轨迹的计算时间，分别包含 7、11 和 21 个门（对应 Split-S、AlphaPilot 和 AirSim）。此外，在包含 5、15、30 和 35 个门的随机生成赛道上评估计算时间。对于每种门数量，运行 3 个不同实验并计算平均计算时间。由于优化方法计算时间过长，对 35 个门的情况仅进行 1 次实验。总共报告了 13 条挑战性竞速赛道的计算时间。

**图 8：** 我们的学习方法与基于优化方法（CPC）之间的计算时间比较。横轴为门数量，纵轴为计算时间（小时）。结果显示基于优化方法的计算时间随门数量急剧增加（因为优化问题中的决策变量数量相应增长），而学习方法的训练时间不依赖于门数量，使该方法对大规模环境特别有吸引力。

#### 2) 消融研究

我们进行消融研究以验证所提方法的设计选择。具体关注不同观测模型和安全奖励的影响。

**观测门数量：** 由于观测中包含的未来门数量既不对应固定距离也不对应固定时间，我们认为有必要消融门数量对策略性能的影响。在 AlphaPilot 赛道上使用 1、2 和 3 个未来门训练策略。使用赛道的两个版本：确定性布局以及第 IV-C 节中有界位移的布局。最终评估模型根据在标准赛道上的最快圈速选择。

**表 IV：门观测消融实验结果。报告标准赛道上的圈速时间以及 1000 条测试赛道上的平均圈速时间和碰撞比率。**

| 门数量 | 圈速时间 | 平均圈速时间 (s) / 碰撞比率 (%) |
|-------|---------|-------------------------------|
| 1 | 8.20 | 9.92 / 23.0 |
| 2 | 8.14 | 8.32 / 2.5 |
| 3 | 8.16 | 8.36 / 2.3 |

结果表明，确定性赛道上的最终圈速时间不受观测门数量的显著影响。这可以解释为任务的静态性质使得即使没有下一个门之后的信息也可以对赛道布局过拟合。然而，引入门位移后，仅使用一个未来门的信息不足以实现快速圈速时间和最小化碰撞比率。

**安全奖励：** 我们引入安全奖励以改变性能目标，使门穿越时到门中心的距离在训练期间被最小化。这在具有显著赛道随机化的训练设置以及因物理系统不确定效应而需要安全裕度的真实世界实验中变得相关。实验设计用于评估两种安全奖励配置对碰撞次数和成功门穿越安全裕度的影响。

**表 V：安全奖励消融实验结果。报告标准赛道布局上的圈速时间、1000 条测试赛道上的碰撞比率以及成功门穿越的安全裕度统计。**

| $d_{max}$ | 圈速时间 (s) | 碰撞比率 (%) | 安全裕度 (m) |
|-----------|-------------|-------------|-------------|
| 0.0 | 8.32 | 2.5 | 0.67 ± 0.28 |
| 2.5 | 8.42 | 0.8 | 0.95 ± 0.34 |
| 5.0 | 8.48 | 0.6 | 0.94 ± 0.35 |

结果表明安全奖励增加了所有配置的平均安全裕度。此外，结果表明安全奖励持续减少了测试集上的碰撞次数。然而，两种安全奖励配置都导致标准赛道上的性能下降。

### F. 真实世界飞行

最后，我们在物理四旋翼上执行策略生成的轨迹，以验证策略的可迁移性。使用的四旋翼由无人机竞速用的商用组件开发，包括碳纤维机架、无刷直流电机、5 英寸螺旋桨和 BetaFlight 飞控。该四旋翼的推重比约为 4。通过在 Split-S 赛道上执行连续四圈的确定性策略展开生成参考轨迹，然后由模型预测控制器（model predictive controller）进行跟踪。图 9 展示了真实世界飞行的合成图像。我们的轨迹使飞行器能够以高达 60 km/h 的速度飞行，将平台推向其物理极限。然而，执行轨迹会产生较大的跟踪误差。未来的研究将专注于改善时间最优轨迹的跟踪性能。

**图 9：** 真实世界飞行的合成图像。

---

## V. 结论

本文提出了一种基于学习的方法，用于训练能够为四旋翼生成穿越多个门的近时间最优轨迹的神经网络策略。我们展示了方法的优势，包括近时间最优性能、处理大规模赛道变化的能力，以及在保持计算效率的同时应对大规模随机赛道布局的可扩展性和泛化能力。我们使用物理四旋翼验证了生成的轨迹，实现了高达 60 km/h 的激进飞行。这些发现表明，深度强化学习在为四旋翼生成自适应时间最优轨迹方面具有潜力，值得进一步研究。

---

## 致谢

作者感谢 Philipp Foehn 和 Angel Romero 在真实世界实验中的帮助以及提供轨迹优化基线。同时感谢亚琛工业大学人机交互研究所的 Jürgen Rossmann 教授和 Alexander Atanasyan 的有益建议。

---

## 参考文献

[1] P. Foehn, A. Romero, and D. Scaramuzza, "Time-optimal planning for quadrotor waypoint flight," Science Robotics, 2021.

[2] G. Loianno, C. Brunner, G. McGrath, and V. Kumar, "Estimation, control, and planning for aggressive flight with a small quadrotor with a single camera and imu," IEEE Rob. and Autom. Letters, 2017.

[3] E. Kaufmann, A. Loquercio, R. Ranftl, M. Müller, V. Koltun, and D. Scaramuzza, "Deep drone acrobatics," RSS: Robotics, Science, and Systems, 2020.

[4] H. Moon, J. Martinez-Carranza, T. Cieslewski, M. Faessler, D. Falanga, A. Simovic, D. Scaramuzza, S. Li, M. Ozo, C. De Wagter et al., "Challenges and implemented technologies used in autonomous drone racing," Intelligent Service Robotics, 2019.

[5] J. A. Cocoma-Ortega and J. Martínez-Carranza, "Towards high-speed localisation for autonomous drone racing," in Mexican International Conference on Artificial Intelligence. Springer, 2019.

[6] E. Kaufmann, M. Gehrig, P. Foehn, R. Ranftl, A. Dosovitskiy, V. Koltun, and D. Scaramuzza, "Beauty and the beast: Optimal methods meet learning for drone racing," in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 690–696.

[7] R. Madaan, N. Gyde, S. Vemprala, M. Brown, K. Nagami, T. Taubner, E. Cristofalo, D. Scaramuzza, M. Schwager, and A. Kapoor, "Airsim drone racing lab," in NeurIPS 2019 Competition and Demonstration Track. PMLR, 2020.

[8] W. Guerra, E. Tal, V. Murali, G. Ryou, and S. Karaman, "FlightGoggles: Photorealistic sensor simulation for perception-driven robotics using photogrammetry and virtual reality," in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.

[9] P. Foehn, D. Brescianini, E. Kaufmann, T. Cieslewski, M. Gehrig, M. Muglikar, and D. Scaramuzza, "Alphapilot: Autonomous drone racing," RSS: Robotics, Science, and Systems, 2020.

[10] G. Ryou, E. Tal, and S. Karaman, "Multi-fidelity black-box optimization for time-optimal quadrotor maneuvers," RSS: Robotics, Science, and Systems, 2020.

[11] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion over challenging terrain," Science robotics, vol. 5, no. 47, 2020.

[12] J. Hwangbo, I. Sa, R. Siegwart, and M. Hutter, "Control of a quadrotor with reinforcement learning," IEEE Robotics and Automation Letters, vol. 2, no. 4, pp. 2096–2103, 2017.

[13] S. Gu, E. Holly, T. Lillicrap, and S. Levine, "Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates," in 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.

[14] S. Karaman and E. Frazzoli, "Sampling-based algorithms for optimal motion planning," The international journal of robotics research, 2011.

[15] D. J. Webb and J. Van Den Berg, "Kinodynamic rrt*: Asymptotically optimal motion planning for robots with linear dynamics," in IEEE International Conference on Robotics and Automation. IEEE, 2013.

[16] R. E. Allen and M. Pavone, "A real-time framework for kinodynamic planning in dynamic environments with application to quadrotor obstacle avoidance," Robotics and Autonomous Systems, vol. 115, pp. 174–193, 2019.

[17] B. Zhou, F. Gao, L. Wang, C. Liu, and S. Shen, "Robust and efficient quadrotor trajectory generation for fast autonomous flight," IEEE Robotics and Automation Letters, 2019.

[18] S. Spedicato and G. Notarstefano, "Minimum-time trajectory generation for quadrotors in constrained environments," IEEE Transactions on Control Systems Technology, vol. 26, no. 4, pp. 1335–1344, 2017.

[19] F. Fuchs, Y. Song, E. Kaufmann, D. Scaramuzza, and P. Duerr, "Super-human performance in gran turismo sport using deep reinforcement learning," IEEE Robotics and Automation Letters, 2021.

[20] M. Jaritz, R. De Charette, M. Toromanoff, E. Perot, and F. Nashashibi, "End-to-end race driving with deep reinforcement learning," in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 2070–2075.

[21] S. Aradi, "Survey of deep reinforcement learning for motion planning of autonomous vehicles," IEEE Transactions on Intelligent Transportation Systems, 2020.

[22] N. Roy and S. Thrun, "Motion planning through policy search," in IEEE/RSJ International Conference on Intelligent Robots and Systems, vol. 3. IEEE, 2002, pp. 2419–2424.

[23] A. Rodriguez-Ramos, C. Sampedro, H. Bavle, P. De La Puente, and P. Campoy, "A deep reinforcement learning strategy for uav autonomous landing on a moving platform," Journal of Intelligent & Robotic Systems, 2019.

[24] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. The MIT Press, 2018.

[25] A. Liniger, A. Domahidi, and M. Morari, "Optimization-based autonomous racing of 1:43 scale rc cars," Optimal Control Applications and Methods, vol. 36, no. 5, pp. 628–647, 2015.

[26] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.

[27] Y. Song, S. Naji, E. Kaufmann, A. Loquercio, and D. Scaramuzza, "Flightmare: A flexible quadrotor simulator," in Conference on Robot Learning (CoRL), 2020.

[28] A. Raffin, A. Hill, M. Ernestus, A. Gleave, A. Kanervisto, and N. Dormann, "Stable baselines3," https://github.com/DLR-RM/stable-baselines3, 2019.

[29] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," in 2011 IEEE international conference on robotics and automation. IEEE, 2011, pp. 2520–2525.
