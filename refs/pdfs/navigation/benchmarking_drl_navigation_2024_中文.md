# 传感器拒止环境中深度强化学习导航的基准测试

**作者：** Mariusz Wisniewski, Paraskevas Chatzithanos, Weisi Guo, Antonios Tsourdos

**单位：** Centre for Autonomous and Cyberphysical Systems, Cranfield University, College Road, Cranfield, MK43 0AL, Bedfordshire, UK.

**通讯作者：** m.wisniewski@cranfield.ac.uk

---

## 摘要

深度强化学习（Deep Reinforcement Learning, DRL）被用于实现未知环境中的自主导航。大多数研究假设传感器数据是完美的，但真实世界环境可能包含自然和人为的传感器噪声与拒止（sensor denial）。本文提出了一项在具有可配置传感器拒止效应的导航任务中，对常用及新兴DRL算法的基准测试。特别地，我们关注不同DRL方法（如无模型的PPO与基于模型的DreamerV3）在传感器拒止条件下受到的影响。结果表明，DreamerV3在具有动态目标的视觉端到端导航任务中优于其他方法——而其他方法无法学会该任务。此外，DreamerV3在传感器拒止环境中总体上也优于其他方法。为了提高鲁棒性，我们使用对抗训练（adversarial training）并证明了在拒止环境中性能的提升，尽管这通常会带来在普通环境中的性能代价。我们期望这一对不同DRL方法的基准测试以及对抗训练的使用，能够成为开发更精密导航策略的起点，使其能够处理不确定和被拒止的传感器读数。

**关键词：** 自主性；导航；传感器融合（Sensor Fusion）；深度强化学习；机器学习；复杂环境；传感器故障（Sensor Faults）；传感器失效（Sensor Failure）；对抗攻击（Adversarial Attacks）；对抗训练（Adversarial Training）；对抗环境扰动（Adversarial Environmental Perturbations）

arXiv:2410.14616v1 [cs.RO] 2024年10月18日

---

## 1 引言

自主导航是无人系统中的一项基本挑战。传统自主系统通常使用同时定位与建图（Simultaneous Localisation and Mapping, SLAM）结合卡尔曼滤波器（Kalman Filter），融合来自多个来源的数据以映射环境、定位智能体并生成到达目标的轨迹。近年来，深度强化学习方法已被用于训练能够使用相机进行导航的端到端策略。这些改进得益于卷积神经网络（Convolutional Neural Networks）在数字图像处理方面的进步（通过反向传播训练）[1-3]，以及使用深度神经网络对RL进行参数化 [4]。

虽然自主视觉导航在仿真环境中是可实现的 [5]，但大多数研究假设传感器读数是完美的。然而在真实场景中，关于环境的信息可能是不确定的、不完整的，甚至是被有意拒止的。这会限制导航系统适应环境变化和做出最优决策的能力。对于安全关键操作（如无人飞行器或自动驾驶车辆），这些失效模式及其对系统的影响需要被充分理解。对于DRL智能体（其本身就难以解释），研究传感器伪影或失效如何影响训练和评估尤为重要。

此外，真实系统通常使用一组传感器。虽然DRL领域的许多前沿工作聚焦于端到端视觉导航，但在实践中理解模型如何跨不同模态学习（即从多个传感器获取数据）非常重要。在本工作中，我们研究传感器失效如何影响使用相机或激光雷达（Lidar）作为观测空间的策略的训练和评估。

为了理解传感器噪声和失效的影响，我们比较了几种DRL架构：TD3、PPO、PPO-LSTM和DreamerV3。TD3和PPO常用于评估连续动作空间的环境，PPO-LSTM是PPO的循环版本（Mirowski等人 [6] 表明循环模型优于非循环模型），DreamerV3则使用自编码器（Autoencoder）创建环境的内部世界模型，并被证明在某些环境中优于PPO。

我们在"普通"环境中（无传感器扰动）生成性能基准，然后逐步向环境中添加传感器受扰的随机区域。随后评估策略，以了解在不同程度噪声下导航的成功率。为了训练和评估这些方法，我们使用了基于ROS的gymnasium环境DRL-Robot-Navigation [7] 的修改版本，其中包含一个3D迷宫，机器人需要导航到目标。

本文是ICUAS 2024会议论文 [8] 关于自主导航研究的扩展。

### 1.1 DRL在导航中应用的综述

多项研究在仿真环境中使用相机作为唯一传感器实现了基于DRL的端到端导航 [9]。常用环境包括DeepMind Lab [10]、ViZDoom [11] 和DRL-Robot-Navigation [7]。

Mirowski等人 [6] 展示了A3C [12] 算法在DeepMind Lab环境中的应用。他们使用相机作为唯一传感器，其最佳架构由图像编码器、2层LSTM、深度预测和回路预测组成。智能体因到达目标或子目标（以水果形式呈现）而获得奖励，不因撞墙而受到惩罚，从而允许"沿墙行走"等不需要记忆的策略。DRL还可以用于FPV无人飞行器与专业人类飞行员的竞速 [13]，通过将图像传感器与IMU结合（用于检测门框并生成VIO状态作为DRL算法的输入）在静态环境中工作。Polvara等人展示了如何使用DQN层次结构实现无人飞行器的自主着陆 [14]。他们的方法处理了大量仿真环境并在某些条件下优于人类飞行员。可解释的深度强化学习方法用于端到端自动驾驶可以处理复杂的城市场景 [15]，其方法在有周围车辆的城市场景中优于多种基线方法，包括DQN、DDPG、TD3和SAC。Pasukonis等人 [16] 评估了3D迷宫中FPV智能体的记忆能力，发现Dreamer和IMPALA等模型在较小迷宫（9x9）上能匹配人类表现，但在较大迷宫（15x15）上性能显著下降。Lample和Chaplot [17] 展示了DRL可用于训练智能体玩第一人称射击游戏DOOM，智能体需要在3D迷宫中导航、收集物品并射击敌人。他们发现使用带LSTM层的DQN和DQRN可以收敛，但需要额外的游戏特征预测（使用单独的损失函数）才能可靠地射击敌人。液态神经网络（Liquid Neural Networks）[18] 近年来被提出用于在变化环境中实现鲁棒导航 [19]。在大多数这些案例中，虽然物理环境是未知的或随机的，但论文未考虑受损的传感器读数（液态神经网络除外，其考虑了噪声和其他扰动）。

### 1.2 攻击下传感器融合的综述

#### 1.2.1 轨迹估计方法

第一种方法处理间歇性攻击或拒止，例如当GNSS覆盖丢失时，使用卡尔曼滤波器（或更高级的循环神经网络）维持轨迹估计直到覆盖恢复 [20, 21]。这对短暂中断是有用的。相同的思路可以预先发现其他V2V网络以改善定位 [22]。然而，如果环境需要持续的急转弯以避开动态物体，则轨迹估计方法不适用。此外，如果攻击是持续的并侵蚀维持精度，则该方法也不适合。

#### 1.2.2 传感器融合与生成式AI方法

另一种方法适用于能够承受多样化传感器套件的高端平台，通过传感器融合来减轻一个传感器被攻破的影响。这通常不适用于低端无人机且不能保证成功 [23]。一种更廉价的替代方案是使用生成式AI利用单个廉价传感器生成替代传感器表示。例如，可以使用相机传感器生成红外图像，然后融合在一起以实现更鲁棒的环境识别 [24]。然而，这些方法需要机载多样化传感器套件或强大的GAN在机载运行，且不能保证成功。

#### 1.2.3 DRL中的对抗环境扰动

第一人称视角（First-Person-View, FPV）导航问题通常被视为部分可观测的 [25]，因为在任何给定时刻，智能体只能看到整个状态的一部分。但传感器故障也必须被考虑。鲁棒部分可观测MDP（Robust POMDP）[26] 认为POMDP具有不确定参数。其他方法 [27] 将对手与智能体之间的交互视为零和极小极大博弈，并将其定义为概率动作鲁棒MDP（Probabilistic Action Robust MDP）。Zhang等人 [28] 将对抗扰动的添加视为状态对抗MDP（State-Adversarial MDP, SA-MDP）。在这种框架下，环境返回的状态可能被对手扰动（对手可能是另一个智能体或扰动状态观测的环境过程），使观测变得不确定，导致智能体返回的动作可能是次优的。因为我们认为对手可以是静态的（即它是环境的一部分），所以我们更倾向于使用SA-MDP框架来描述我们的问题。

在构建鲁棒RL模型时，一些工作关注物理扰动 [29]（可能影响智能体的物理特性，如引入不确定力），而另一些则考虑对传感器状态的扰动 [30]。Korkmaz [31, 32] 认为普通训练比最先进的对抗和鲁棒训练技术产生更鲁棒的策略。Havens等人 [33] 提出了一种在对手存在时切换子策略的层次RL方法。对抗训练的应用 [34] 展示了如何使用零和动作鲁棒RL方法来减少空中交通管理场景中的冲突。

### 1.3 研究空白与创新点

尽管DRL和传感器攻击的研究已经存在，但我们没有找到将传感器失效对DRL学习导航策略性能影响进行详尽研究的工作。我们旨在理解传感器失效——自主系统中一种潜在灾难性的失效模式——对DRL智能体学习导航策略能力的影响。为此我们：

- 在DRL-Robot-Navigation环境上对不同强化学习算法进行基准测试。
- 研究观测空间（激光雷达和相机）对导航能力的影响。
- 进行定量和定性研究，展示对抗扰动（噪声和传感器失效）对DRL算法在导航任务中性能的影响。

此外，为了使他人能够复现我们的工作，我们开源了对DRL-Robot-Navigation环境的修改，并将其更新为符合gymnasium [35] 接口。^1

本文在第2节中解释方法论，包括环境、环境的变更以及强化学习模型的选择；在第3节中展示不同算法在环境上的基准测试结果、激光雷达和相机噪声研究，以及与文献的讨论；最后在第4节中总结并提出后续工作建议。

> ^1 https://github.com/mazqtpopx/cranfield-navigation-gym

---

## 2 方法论

我们基于DRL-Robot-Navigation环境 [7] 进行构建，该环境模拟机器人在一个10m x 10m的迷宫中导航，迷宫中分布有静态障碍物。机器人配备一组传感器——激光雷达、相机和里程计——但原始论文中仅使用其中一部分作为状态观测来训练强化学习模型。原始论文使用激光雷达、到目标的距离和到目标的角度拼接作为观测空间，不使用相机。环境示意如图1所示，其中展示了DRL-Robot-Navigation环境及传感器读数。

由于本研究关注对抗传感器扰动和测试不同传感器套件，环境的变更在第2.1节描述。基准测试流程（使用不同模态和不同传感器噪声训练不同RL算法——TD3、PPO、PPO LSTM和DreamerV3）在图2中说明。基准测试中使用的强化学习算法在第2.2节描述。训练实验设置（包括超参数）在第2.3节展示。评估过程在第2.4节描述。

### 2.1 DRL-Robot-Navigation环境的变更

为了支持我们的实验，DRL-Robot-Navigation环境需要进行若干修改。一项主要变更是添加基于视觉的观测空间：将机器人的图像从环境中输出，并添加视觉标记来标识目标位置。在目标所在位置添加一个直径0.5m的粉色球体，如图1（左上图）所示。

碰撞系统进行了更新，因为原始实现依赖于激光雷达传感器的读数，这不是检测碰撞的可靠方式，且向激光雷达传感器添加噪声可能触发碰撞。相反，我们通过触发可订阅的ROS话题来生成标志并检测对象之间的任何物理碰撞。

我们添加了包含传感器噪声的区域：分别用于激光雷达和相机。这些区域是固定大小的矩形，在每个回合开始时在地图的随机位置生成。传感器故障的建模方式在第2.1.1节进一步描述。

我们更新了奖励系统。最初，智能体到达目标获得100奖励，与墙壁碰撞获得-100。否则，返回动作函数（惩罚机器人不动）和与墙壁最小距离（基于激光雷达观测，当机器人距墙不到1m时）的函数值。首先，我们将奖励归一化到+1和-1之间。然后，我们添加了每步惩罚：大小为-1除以最大步数。这样做是为了防止机器人不移动的局部最小值，并与其他gymnasium环境 [35] 保持一致。我们还认为惩罚不动和激光雷达距墙距离的函数是不必要的。相反，我们认为这些项——不动和避障——应该通过策略端从观测中学习，而非通过奖励塑形。

#### 2.1.1 对抗传感器扰动建模

自动驾驶车辆经常使用相机作为导航的主要传感器，但物理传感器攻击很少被考虑。Kim等人 [36] 对无人机攻击进行了综述，包括对相机传感器的激光攻击，发现当施加攻击时，相机像素会失真。这表明相机传感器易受物理攻击。相机攻击被建模为完全失效，即在相机噪声矩形区域内所有像素变黑（0）。虽然这并非明确模仿物理攻击，但它是相机的一种可能的临时失效模式，如图3(b)所示。激光雷达攻击通过向每个激光雷达点添加0-5米范围的强高斯噪声来建模。传感器噪声在噪声区域内均匀施加。激光雷达传感器扰动如图3(a)所示。

> **图1：** DRL-Robot-Nav迷宫环境示例 [7]。左上：Gazebo渲染的环境俯视图。左下：RGB相机视图。右：RViz可视化的离散激光雷达传感器读数、机器人和噪声传感器区域（蓝色：相机噪声区域，红色：激光雷达噪声区域）。粉色球体（左上）作为视觉标记标识基于视觉策略的目标。

> **图2：** 实验流程图，展示所提实验的过程。环境包含干净的、相机噪声和激光雷达噪声变体。环境以两种模态（激光雷达和相机）输出观测。然后训练多种算法在避开障碍物的同时导航到目标。动作空间：前进 [0,1]，左转/右转 [-1, 1]。

> **图3：** (a) 机器人在激光雷达噪声区域内的RViz可视化示例。进入红色激光雷达噪声区域后，离散激光雷达点被添加高斯噪声，不再提供环境的准确表示。(b) 机器人在相机噪声区域内的RViz可视化示例。进入蓝色相机噪声区域后，图像馈送（左侧所示）的所有像素变黑——模拟相机失效。

### 2.2 强化学习基准测试

在DRL-Robot-Navigation环境上对以下RL方法进行了基准测试：

- **TD3（Twin Delayed DDPG）**[37]：一种离线策略（off-policy）、无模型（model-free）算法，设计用于连续动作空间。它采用两个Q值近似器（评论家）并使用较小的Q值来更新策略，这有助于缓解过高估计偏差（overestimation bias）问题。TD3还引入了策略网络更新的延迟以减少策略更新的方差，并向目标动作添加噪声以促进探索。TD3的训练超参数见表1。

- **PPO（Proximal Policy Optimization，近端策略优化）**[38]：一种在线策略（on-policy）、无模型算法，设计用于连续空间。还使用了循环PPO（Recurrent PPO），即引入长短期记忆（Long Short-Term Memory, LSTM）层 [39] 的PPO修改版本。PPO和循环PPO的训练超参数见表2。循环架构与PPO类似，但在Actor-Critic多层感知器之前（对于相机观测则在卷积层之后）包含一个LSTM层。

- **DreamerV3** [40]：一种离线策略、基于模型（model-based）的算法，通过结合自编码器将输入编码为离散表示来从经验中学习世界模型。它使用循环结构并预测动态、奖励和回合的延续性。其超参数是固定的，因此不需要调参。它已被证明在多个环境中优于PPO，包括在DeepMind Lab [10] 上的导航。基于此，预期DreamerV3应在我们的导航问题上优于其他算法。使用默认模型。

实验使用了Stable-Baselines3 [41] 的TD3、PPO和循环PPO实现。TD3和PPO的一般设置如图4所示，展示了相机和激光雷达观测的网络结构。对于相机观测，图像首先通过卷积网络将维度降低到256个特征，然后这些特征被传入actor-critic MLP。对于循环PPO，特征首先通过LSTM层，再进入actor-critic。对于激光雷达，因为仅包含24个数据点，特征直接传入actor-critic。我们执行：

- 对RL方法的基准测试。
- 学习率搜索以找到每种算法的最优学习率，因为它是最敏感的超参数（DreamerV3除外，因为它不需要超参数调优）。详见附录。
- 对智能体在不同观测空间下在传感器扰动区域中性能的研究。

> **图4：** 相机和激光雷达观测网络。在相机观测模型（TD3、PPO、PPO-LSTM）中，图像首先通过卷积层并展平为256个特征以降低输入维度。这些特征随后输入actor-critic MLP。对于激光雷达观测模型（TD3、PPO），输入本身就是低维的，因此20个激光雷达读数与到目标距离、到目标角度和先前动作拼接后输入MLP。

> **表1：** TD3训练参数（最终值加粗）
>
> | 参数 | 值 |
> |------|------|
> | 学习率 | 0.03 / 0.003 / **0.0003** / 0.00003（见LR研究） |
> | Tau | 0.005 |
> | 学习开始（步数之后） | 5000 |
> | 批大小 | 16384（激光雷达）, 256（相机） |
> | 折扣因子 | 0.99 |
> | 探索噪声 | 0.3 |

> **表2：** PPO和PPO LSTM训练参数（最终值加粗）
>
> | 参数 | 值 |
> |------|------|
> | 学习率（激光雷达） | 0.03 / **0.003** / 0.0003 / 0.00003（见LR研究） |
> | 学习率（相机） | 0.03 / 0.003 / **0.0003** / 0.00003（见LR研究） |
> | 批大小 | 256 |
> | 折扣因子 | 0.99 |

### 2.3 实验场景设置

实验设置如表3所示。激光雷达观测空间包含20个离散点，在机器人周围180°弧线上等间隔采样。相机图像产生160x160像素的RGB图像，DreamerV3使用64x64像素的RGB图像（DreamerV3的输入需要是2的幂次^2）。激光雷达噪声为高斯噪声，相机噪声导致完全黑屏。智能体位置输入（仅限激光雷达模态）完全准确。激光雷达的目标位置是随机的，相机的是固定的。步间时间间隔为0.05秒——这是智能体执行动作之间的延迟。注意仿真以4.0倍实时速度运行，使得该时间步为0.2秒，即智能体实际上以5 Hz的频率运行。奖励函数在智能体到达目标时产生稀疏奖励，碰撞墙壁时产生惩罚，每步有一个小惩罚以防止不动。每回合最大步数为50。

> ^2 https://github.com/danijar/dreamerv3/issues/12

> **表3：** 仿真参数
>
> | 参数 | 详情 |
> |------|------|
> | 激光雷达测量 | 180° LiDAR数据离散化为20个区间 |
> | 相机测量 | 150px x 150px RGB图像；64px x 64px RGB图像（Dreamer） |
> | 攻击类型（激光雷达） | 向每个离散点添加高斯噪声 |
> | 攻击类型（相机） | 完全失效 |
> | 智能体位置 | 完全准确（来自仿真的真值） |
> | 目标位置 | 激光雷达：随机；相机：固定 |
> | 步间延迟 | 0.05秒 |
> | 奖励函数 | $R(s, a) = \begin{cases} 1, & \text{到达目标} \\ -1, & \text{碰撞} \\ -1/50, & \text{其他} \end{cases}$ |
> | 最大回合步数 | 50 |

仿真中存在若干影响学习过程的随机参数。如Mirowski等人 [6] 所示，随机性起着关键作用。在他们的仿真中，RL智能体在静态小迷宫上优于人类，但在大型复杂迷宫（具有随机参数如随机智能体生成）上表现困难。DRL-Robot-Navigation与DeepMind Lab的另一个区别是我们的环境会惩罚撞墙，从而阻止沿墙行走策略。

相机策略比激光雷达需要更长的训练时间（超过500k步），超出了本研究的计算预算。为简化，目标生成在地图中心而非随机生成。机器人仍以随机位置和方向生成。这类似于DeepMind Lab中Mirowski等人测试的"静态迷宫"。他们的"随机目标迷宫"类似于我们在激光雷达实验中测试的随机目标位置（尽管DeepMind Lab不惩罚撞墙）。

通常，除DreamerV3外，相机策略可能需要更长时间收敛，因为观测空间更大：即使经过卷积层，状态仍包含256个特征，而激光雷达仅有24个点（包括先前动作、到目标距离和到目标角度）。

DRL-Robot-Navigation环境的原始实现在每回合开始时随机化机器人的起始位置和方向，以及目标位置。

为了改善相机策略的收敛性以便更好地理解传感器拒止效应，我们为相机实验简化了环境。我们将目标位置固定在迷宫中心。这减少了环境中的随机性，使策略能够快速收敛。对于激光雷达实验，我们保持目标和机器人的生成位置为随机。

### 2.4 训练与评估

第2.2节中描述的强化学习模型使用以下配置和环境进行训练：

- **激光雷达**（包含到目标距离和角度）：模型在0x0（无噪声）地图以及具有不同噪声量的地图上训练：3x3、5x5和7x7米噪声区域。
- **仅相机，固定目标**：模型在0x0（无噪声）地图以及具有不同噪声量的地图上训练：3x3、5x5和7x7米噪声区域。

除非另有说明，相机配置包含固定目标以实现更好的收敛，如第2.3节所述。然后在第2.4节指定的地图上评估每个模型，通常包括在不同噪声大小的环境上进行测试：0x0、3x3、5x5、7x7。在某些情况下，模型在简化版地图上评估，如第2.4节所述。

图5展示了激光雷达和相机环境的训练和评估设置概要。

> **图5：** 激光雷达和相机传感器的训练与评估场景。对于激光雷达场景，模型在0x0（普通）地图和具有不同传感器扰动量的地图上训练：3x3、5x5和7x7米传感器噪声区域。这些区域包含添加到传感器读数的高斯噪声。训练后的策略在相同地图上评估。类似地，对于相机场景，模型在0x0（普通）地图和具有不同传感器扰动量的地图上训练：3x3、5x5和7x7米传感器拒止区域。在这些区域内，相机传感器完全失效（所有像素变黑）。模型在默认地图上训练，目标固定在中间，蓝色区域为相机完全失效区域。策略在不同噪声程度的默认地图上评估。第二次评估在无障碍物地图上进行，以更好地理解传感器拒止区域中的学习策略。

#### 2.4.1 相机评估地图

对于相机策略的评估，使用两种不同的地图：

- DRL-Robot-Navigation地图（与训练相同）。
- 简化的DRL-Robot-Navigation地图（无障碍物），如图6所示。目标始终固定在中心。噪声区域固定在地图中心。

使用简化地图的目的是更好地理解策略的行为。由于移除了障碍物，唯一剩下的任务是穿过传感器拒止区域到达目标。由于噪声是固定的，它始终遮挡目标。在这个评估任务中，我们忽略智能体避障的能力，而关注智能体是否学会了穿过噪声区域。简化地图仅用于评估，从不用于训练策略。默认使用DRL-Robot-Navigation进行评估。简化地图的使用在结果部分明确说明。

> **图6：** DRL-Robot-Navigation地图的简化版本，无障碍物，目标固定在中心，传感器扰动区域围绕目标。(a) 3x3传感器拒止区域；(b) 5x5传感器拒止区域；(c) 7x7传感器拒止区域。机器人在地图周围以随机位置和方向生成。该地图仅用于评估相机策略，不用于激光雷达策略。

---

## 3 结果

进行了多项实验以对不同DRL算法在环境上进行基准测试，并理解不同训练方案（普通 vs. 对抗）如何影响模型在不同环境扰动程度下的性能。虽然第3.1节首先比较了激光雷达和相机模态的训练，但其余结果部分根据模态分为两部分。第一部分在第3.2节中展示PPO和TD3的定量结果，然后在第3.3节中通过理解策略所走路径及其在不同训练方案下的差异进行定性分析。

相机观测部分从第3.4节不同算法在普通环境上的直接训练比较开始，在第3.5节继续进行带传感器拒止的不同模型定量基准测试，并在第3.6节在无障碍物地图上评估模型。最后，在第3.7节对结果进行文献讨论。

### 3.1 模态研究

图7所示的模态研究展示了PPO算法在激光雷达（随机目标）、固定目标相机和随机目标相机之间的模型性能差异。激光雷达和固定目标相机能够快速收敛到接近最大回合奖励的解。随机目标相机在给定步数内未能收敛到解。由于我们试图理解传感器扰动的影响，我们将继续使用固定目标相机进行相机噪声实验。

> **图7：** 使用PPO的模态研究（激光雷达和相机）。相机在固定和随机目标上训练。每个模型训练500,000步。图中展示了激光雷达快速收敛、固定目标相机也快速收敛，但随机目标相机未收敛。

### 3.2 带噪声的激光雷达导航

虽然第2.2节提出了不同算法，但我们无法找到合适的学习率使PPO-LSTM收敛（如附录A3所示）。虽然可以在激光雷达观测空间上训练DreamerV3，但对于如此小的观测空间，它是一个大模型，且我们没有发现其性能优于其他模型。因此本节仅比较TD3和PPO。

图8展示了使用激光雷达观测空间的TD3和PPO在默认地图上的评估结果。算法在0x0（普通）地图上训练，在激光雷达噪声区域递增的环境上评估：0x0（普通）、3x3、5x5和7x7——指噪声区域矩形的米数大小。TD3在不同评估中均优于PPO。它在0x0评估场景中达到87.2%的平均成功率，而PPO为83.0%。两个模型之间的性能差距通常随噪声区域增大而扩大。在7x7噪声区域评估时，TD3达到51.4%，而PPO为35.2%。

> **图8：** 在默认无噪声环境上训练的算法的激光雷达噪声区域研究。然后在不同噪声区域大小的默认环境（随机目标）上评估。评估进行100回合并重复5次以建立标准差。

图9展示了在3x3噪声地图上训练的TD3和PPO的相同评估结果。在这些结果中，PPO在各评估中与TD3性能接近匹配，在较大噪声区域上超越TD3。在普通0x0场景评估时，TD3达到69.6%的平均成功率，PPO达到68.0%。这相比在普通0x0环境上训练的模型有性能下降。在7x7场景评估时，PPO模型达到54.0%的平均成功率。这相比在普通0x0环境上训练的PPO模型有所改善，并略微优于在相同方案下训练的TD3模型。在3x3噪声环境上训练的TD3模型在所有评估场景中相比在0x0环境上训练的TD3模型都出现了性能下降。

> **图9：** 在默认环境上训练（含随机3x3噪声区域和随机目标）的算法的激光雷达噪声区域研究。然后在不同噪声区域大小的相同地图上评估。评估进行100回合并重复5次以建立标准差。

### 3.3 激光雷达噪声训练方案的定性结果

如第3.2节所示，训练方案影响算法的评估性能。为了更好地理解激光雷达特定策略的行为，我们可以更仔细地调查在噪声方案中训练对每种算法的影响，并研究评估过程中机器人的轨迹。

图10展示了在不同激光雷达噪声方案上训练的PPO策略在默认地图上的评估。结果表明，普通PPO（0x0）在无噪声地图上评估时性能最好，平均成功率为83%。在噪声环境中训练的策略（PPO 3x3、5x5和7x7）在5x5噪声地图上评估时开始优于普通策略。虽然噪声策略在添加更多噪声时性能未显著下降，但在0x0无噪声环境上评估时，与普通模型相比表现出一致的性能下降。这可能是因为噪声策略采取了更冒险的策略，牺牲避碰能力以提高导航性能。为了更好地理解这一点，我们可以通过绘制路径、碰撞和成功导航来定性地调查评估运行。

> **图10：** 在不同激光雷达噪声方案上训练的PPO策略在默认地图（随机目标）上的评估，使用不同噪声区域大小。策略评估100步5次，图中展示5次重复运行的均值和标准误差。

图11展示了2种不同策略（PPO 0x0和PPO 3x3）在两种不同评估地图（0x0和7x7）上评估时生成的100条路径图。图11(a)展示了普通PPO在无噪声地图上的评估。同一策略在图11(b)中在带随机7x7噪声区域的地图上评估。策略无法到达目标，性能显著下降（从79次成功降至26次成功）。还显示在一半的运行中，策略达到了最大回合时间步限制，既未碰撞也未到达目标。与无噪声地图上的评估相比，路径趋于更短，表明策略在噪声区域内倾向于采取"安全"策略——不采取任何行动。相比之下，当在少量噪声（3x3随机区域）上训练并在无噪声上评估时（图11(c)），总体性能从79次成功下降到69次，碰撞从21次增加到31次。但在7x7噪声区域评估时（图11(d)），策略不再采取"安全"策略，而是选择"更冒险"的方法。这将成功回合从26增加到49，但碰撞也从24增加到51。

总体而言，安全策略并未带来更好的总体性能，因为达到最大步数的奖励是-1，与碰撞的-1相同。但由于冒险策略导致更多成功运行获得+1奖励，它对噪声评估地图来说是更优的策略。有趣的是，当在噪声地图上训练时，PPO运行中一致出现了更冒险的策略，而在未经噪声训练时，策略在传感器读数不确定时采取了更安全的策略让回合耗尽，如图10所示。这可能是因为噪声传感器读数落在了未经噪声环境训练的策略先前观察到的域之外。

> **图11：** 评估过程中生成的100条路径图。(a) 普通策略（PPO 0x0）在0x0噪声地图上评估：79次成功，21次碰撞。(b) 普通策略（PPO 0x0）在7x7噪声地图上评估：26次成功，24次碰撞。(c) 噪声策略（PPO 3x3）在0x0噪声地图上评估：69次成功，31次碰撞。(d) 噪声策略（PPO 3x3）在7x7噪声地图上评估：49次成功，51次碰撞。红点表示碰撞，绿点表示成功导航到目标。

图12同样展示了TD3策略。与PPO训练方案不同，TD3通常没有从在噪声场景上训练中获益，在无噪声地图上训练的默认策略在所有评估场景中表现最好。

> **图12：** 在不同激光雷达噪声方案上训练的TD3策略在默认地图（随机目标）上的评估。策略评估100步5次，图中展示均值和标准误差。

### 3.4 算法训练比较（相机观测）

如第2.2节所述，在DRL-Robot-Navigation环境上训练了不同的RL方法：TD3、PPO、循环PPO（PPO-LSTM）和DreamerV3。

图13展示了不同算法在500,000步训练期间的平均回合奖励。对于固定目标场景，TD3、PPO和PPO-LSTM快速收敛到解，每个都在不到100,000步内达到0.5平均奖励。唯一能够学会导航到随机目标的模型是DreamerV3。相比之下，PPO即使在500,000步后也未能学会导航。尽管如此，为了更好地理解噪声对训练的影响，相机策略将使用固定目标。

> **图13：** 训练期间不同算法收敛的比较。DreamerV3以随机目标训练，而其他模型以固定目标在中心训练（使DreamerV3的收敛更加困难）。

### 3.5 带传感器拒止的相机导航

图14展示了使用相机观测空间的不同算法（TD3、PPO、PPO-LSTM和DreamerV3）的评估结果。算法在普通0x0地图上训练，在相机拒止区域递增的环境上评估：0x0（无传感器拒止）、3x3、5x5和7x7——指传感器拒止区域矩形的米数大小。在0x0评估场景中，DreamerV3达到最高平均成功率90.8%，TD3达到类似的88.8%。PPO-LSTM第三，平均82.2%，PPO表现最差，平均71.6%。在3x3评估场景中，模式保持不变，DreamerV3达到最高平均73.2%（标准误差大得多，为6.8），TD3紧随其后68.4%，PPO-LSTM第三，PPO最后。5x5和7x7评估场景显示所有算法性能严重下降。

> **图14：** 在普通无传感器拒止环境上训练的算法的相机噪声区域研究。然后在不同噪声区域大小的普通环境上评估。评估进行100回合并重复5次以建立标准差。各算法在0x0/3x3/5x5/7x7下的成功率：TD3为88.8/68.4/17.2/1.6%，PPO为71.6/51.4/14.0/1.8%，PPO-LSTM为82.2/60.2/21.0/1.6%，DreamerV3为90.8/73.2/29.0/3.0%。

类似地，图15展示了在带随机3x3拒止区域地图上训练的算法评估结果。在0x0场景评估时，DreamerV3仍达到最高平均成功率86.4%（相比干净场景训练的90.8%有所下降）。TD3第二82.4%，PPO和PPO-LSTM表现最差，分别为71.0%和72.4%。所有策略相比在干净（无传感器拒止区域）环境上训练的策略都出现了性能下降。在3x3场景评估时，DreamerV3表现最好，达到80.6%平均成功率。这相比普通模型（73.2%）有所改善。TD3平均成功率73.2%，PPO为69.2%，PPO-LSTM最后58.6%。同样，TD3和PPO也显示了相比普通模型的改善。PPO-LSTM是唯一未报告比普通模型改善的模型。

所有模型在5x5和7x7噪声区域地图上的评估性能有所提升（相比在无噪声环境中训练的模型），表明在场景中训练一些噪声通常能提高在噪声评估场景中的评估性能。然而，这些改善不够显著，模型不能可靠地到达目标。在3x3场景中使用传感器拒止区域训练模型的缺点是，这些模型在0x0场景评估时似乎会出现性能退化。

> **图15：** 在默认环境（含3x3传感器拒止区域）上训练的算法的相机噪声区域研究。然后在不同传感器拒止区域大小的默认环境上评估。评估进行100回合并重复5次以展示均值和标准误差。各算法在0x0/3x3/5x5/7x7下的成功率：TD3为82.4/73.2/45.0/12.2%，PPO为71.0/69.2/41.8/13.6%，PPO-LSTM为72.4/58.6/36.6/12.2%，DreamerV3为86.4/80.6/40.0/15.0%。

### 3.6 带传感器拒止的相机导航（无障碍物地图）

第3.5节表明训练方案对传感器拒止区域中的策略行为有影响。为了更好地理解这种行为，在简化的无障碍物地图上进行评估，如第2.4节所述。

图16展示了使用相机观测空间的不同算法的评估结果。算法在普通0x0（无传感器拒止）默认地图上训练，在简化的无障碍物地图上评估，传感器拒止区域位于地图中心并逐渐增大：0x0（无传感器拒止）、3x3、5x5和7x7。四个模型都表现出类似行为：在0x0评估中几乎所有回合都成功导航，但当引入任何噪声时一致失败（少数成功可能是由于生成在目标附近）。

> **图16：** 在默认环境（0x0噪声）上训练的算法的相机噪声区域研究。然后在不同噪声区域大小的简化地图上评估。评估进行100回合。在0x0上所有算法接近99-94%，但引入噪声后骤降至0-4%。

类似地，图17展示了在默认地图（含随机3x3噪声区域）上训练的不同算法的评估结果。0x0评估没有显著变化，TD3、PPO和DreamerV3仍分别成功导航98、98和96次（共100次）。对于3x3评估，TD3到达目标74次，PPO 71次，PPO-LSTM 66次，DreamerV3表现最好87次。这相比在0x0（无噪声）环境上训练的模型有显著改善——表明策略已学会在存在噪声时导航到目标。3x3训练的模型在5x5和7x7场景评估时也显示了相比0x0训练模型的性能改善。然而，它们没有达到高（90+）的成功率，这才能表明策略真正解决了场景。

> **图17：** 在默认环境（含3x3噪声）上训练的算法的相机噪声区域研究。然后在不同噪声区域大小的简化地图上评估。评估进行100回合。各算法在0x0/3x3/5x5/7x7下的成功率：TD3为98/74/14/13%，PPO为98/71/38/10%，PPO-LSTM为84/66/49/19%，DreamerV3为96/87/50/20%。

图18展示了在不同相机噪声方案上训练的PPO策略在无障碍物地图上的评估。PPO（0x0）和PPO（3x3）在0x0场景评估时表现最好，分别达到99和98次成功回合。PPO（0x0）在任何噪声地图上都无法导航到目标。PPO（5x5）和PPO（7x7）在0x0评估中都未能成功导航，这表明过多噪声阻碍了这些模型的学习。

> **图18：** 在不同噪声方案上训练的PPO策略在简化地图上的评估，使用不同噪声区域大小。

图19展示了2种不同策略（PPO 0x0和PPO 3x3）在两种不同评估地图（0x0和3x3固定传感器拒止区域、固定目标在中心、无障碍物）上评估时生成的100条路径图。图19(a)展示了普通PPO在0x0地图上的评估。同一普通策略在图19(b)中在目标周围带固定3x3噪声区域的地图上评估。策略无法导航到目标，性能显著下降（从100次成功降至5次成功）。当机器人进入噪声区域时，相机失效，机器人停止移动。这可以从许多路径在蓝色噪声区域内结束且未继续中看出。

相比之下，当在环境中存在3x3随机区域进行对抗训练并在3x3地图上评估时（图19(d)），成功回合增加到75。在蓝色传感器拒止区域内，智能体只是绕圈移动，这可以从蓝色区域中的路径轨迹看出。这不是在环境中存在障碍物时的可靠导航方式，但在传感器拒止区域内无障碍物时能够成功。策略利用了场景的奖励结构，选择了穿过传感器拒止区域的高风险策略。因此，与图11所示的激光雷达示例类似，策略已学会了在传感器读数不确定时盲目导航的高风险策略。

由于我们在简化地图上评估，不存在障碍物。显然，一旦进入传感器拒止区域，策略就无法避障，这在这些结果中被故意不反映，以便理解传感器拒止区域中的策略行为。然而，由于模型在默认地图上训练，它们已学会在存在障碍物的传感器拒止区域中采取冒险行动。这可能意味着由于环境的奖励塑形，策略采取高风险行动更有回报。但这仅在环境中传感器拒止区域较少时成立（如图18所示），因为大量噪声会导致基本导航行为的学习失败。

> **图19：** 评估过程中生成的100条路径图。(a) 普通策略（PPO 0x0）在0x0地图上评估：100次成功，0次碰撞。(b) 普通策略（PPO 0x0）在3x3传感器拒止地图上评估：5次成功，0次碰撞。(c) 对抗训练（PPO 3x3）在0x0传感器拒止地图上评估：98次成功，1次碰撞。(d) 对抗训练（PPO 3x3）在3x3传感器拒止地图上评估：75次成功，1次碰撞。红点表示碰撞，绿点表示成功导航。蓝色区域为传感器拒止区域。

### 3.7 讨论

相机导航中总体表现最好的是DreamerV3。DreamerV3在0x0和3x3噪声区域评估中，无论在0x0还是3x3环境上训练，都一致优于其他模型的相机观测（图14和15）。这可能源于DreamerV3与其他模型具有不同的结构，并通过自编码器结构构建世界模型。我们假设由于自编码器/循环结构，它可能能够克服传感器拒止区域。虽然在3x3环境上训练和评估时它确实优于其他模型（在图14中达到80.6%），但仍未达到在无噪声环境上训练和评估时90.8%的基线性能。DreamerV3也是唯一在500,000步内使用仅相机观测空间学会导航到随机目标的模型（第3.4节）。其他模型要么需要激光雷达策略（包含到目标的距离和角度信息），要么需要固定目标才能使用相机观测空间收敛到解。DreamerV3使用也很方便，因为它不需要任何超参数调优。激光雷达模态中表现最好的（在TD3和PPO之间）是TD3。在0x0上训练的TD3模型在不同评估中通常优于大多数PPO变体。

在第3.2节和第3.6节中，我们了解到在环境中存在一些噪声的情况下训练策略会产生一个有趣的结果：这些策略在导航到目标方面更成功，但在传感器读数不确定或完全拒止时也会采取更多高风险行动，这可能导致更频繁地与障碍物碰撞——在我们的场景中是终止事件。这是在安全关键场景中训练RL策略的一个局限性。训练它们以最大化奖励并在不确定场景中到达目标，可能导致策略学会选择被认为有回报的动作，即使这些动作可能导致灾难性场景。

Korkmaz [31] 报告称对抗训练的策略比普通训练的策略更容易受到亮度或模糊等变化的影响。虽然我们没有考虑对抗学习算法（Korkmaz使用基于SA-MDP和RADIAL算法的SA-DDQN，而我们只是在具有对抗扰动的环境中训练模型），但这可以映射到我们的问题中：噪声训练的策略（如3x3 PPO）导致更多碰撞，在0x0评估中不如普通PPO（0x0）鲁棒。

Mirowski等人 [6] 进行了类似的研究，包括静态迷宫和大型随机迷宫。这分别等同于固定目标（我们使用相机进行）和随机目标（我们使用激光雷达进行）。在他们的论文中，使用循环的算法优于非循环算法。我们在结果中没有看到这种趋势，PPO-LSTM通常表现不佳（忽略同样是循环的DreamerV3）。这令人惊讶，因为导航是一个部分可观测马尔可夫决策过程（POMDP），因此策略应该从记忆中获益。

---

## 4 结论与后续工作

本研究展示了常见强化学习算法在DRL-Robot-Navigation环境中使用不同模态（相机和激光雷达）的定量结果，以及在包含噪声和传感器拒止区域的环境中训练和评估的不同效果。我们观察到，在噪声环境中训练DRL模型会产生在噪声环境中通常表现更好的策略（与在普通无噪声环境中训练的模型相比）。然而，我们发现当面临噪声或故障观测时，这些策略选择了更冒险的行动：可能导致成功导航到目标，但也可能导致与其他物体碰撞。虽然这些模型在定量指标上产生了更好的性能，但从安全关键场景的角度来看，这种行为是不可接受的，表明需要更鲁棒的RL方法来抵御场景中的噪声或传感器中断。因此，我们开源了在DRL-Robot-Navigation之上构建的环境，使其他研究人员能够在这些场景上评估其模型性能。这里展示的结果应作为未来关于如何处理故障传感器策略研究的基准。虽然我们展示了对激光雷达高斯噪声和相机中断的评估，但还应集成不同的扰动（例如 [32] 研究了相机观测的压缩伪影、亮度对比度和模糊等变换）。我们认为自编码器方面的工作可能导致识别传感器失效，并可能作为在传感器中断情况下填补空白的备份。还应研究传感器融合策略。层次强化学习可能带来更好的性能（如Havens等人 [33] 所提出），因为它允许顶层层次检测对手。这可能在有噪声的任务中带来更好的性能，因为局部目标可能在回合过程中改变——从导航到目标变为脱离噪声区域。

最后，我们建议对环境进行进一步开发，以包含环境交互以及更精细的传感器攻击模型。

---

## 附录 A 学习率敏感性

由于算法（除DreamerV3外）对学习率敏感，我们首先进行学习率研究——然后在整个过程中使用该学习率。

> **图A1：** TD3学习率搜索（激光雷达）。展示了lr=0.003、lr=0.0003和lr=0.00003的训练曲线。0.0003和0.00003均收敛到约0.5回合奖励的最大值，但0.0003收敛更快。

图A2展示了PPO算法使用激光雷达观测空间的学习率搜索。0.003似乎是最优值，收敛速度快于0.0003。

> **图A2：** PPO学习率搜索（激光雷达）。展示了lr=0.03、lr=0.003、lr=0.0003和lr=0.00003的训练曲线。

图A3展示了PPO LSTM算法使用激光雷达观测空间的学习率搜索。未实现明确收敛——这令人惊讶，因为我们预期循环网络在该问题上有效。不带循环的PPO在激光雷达观测空间上表现好得多（图A2）。

> **图A3：** PPO LSTM学习率搜索（激光雷达）。展示了lr=0.03、lr=0.003、lr=0.0003和lr=0.00003的训练曲线。未有明确收敛。

图A4展示了PPO LSTM算法使用相机观测空间的学习率搜索。0.0003是唯一收敛到解的值。我们将在PPO和PPO LSTM的相机运行中使用0.0003。

> **图A4：** PPO LSTM学习率搜索（相机）。展示了lr=0.03、lr=0.003、lr=0.0003和lr=0.00003的训练曲线。仅0.0003收敛。

**补充信息：** 修改后的DRL-Robot-Navigation环境可在以下地址获取：https://github.com/mazqtpopx/cranfield-navigation-gym

---

## 致谢

MW和PC的工作由Leonardo UK与Cranfield University资助，WG和AT的工作由EPSRC TAS-S: Trustworthy Autonomous Systems: Security (EP/V026763/1) 资助。

## 声明

- **资助：** MW和PC的工作由Leonardo UK与Cranfield University资助，WG和AT的工作由EPSRC TAS-S: Trustworthy Autonomous Systems: Security (EP/V026763/1) 资助。
- **利益冲突/竞争利益：** 无
- **伦理审批与参与同意：** 无
- **出版同意：** 无
- **数据可用性：** 无
- **材料可用性：** 无
- **代码可用性：** 代码可在 https://github.com/mazqtpopx/cranfield-navigation-gym 获取。
- **作者贡献：** 所有作者对研究构思和设计做出了贡献。材料准备、数据收集和分析由MW和PC完成。手稿初稿由MW撰写，所有作者对手稿的先前版本进行了评论。所有作者阅读并批准了最终手稿。

---

## 参考文献

[1] Krizhevsky, A., Sutskever, I., Hinton, G.E.: ImageNet Classification with Deep Convolutional Neural Networks. In: Advances in Neural Information Processing Systems, vol. 25. Curran Associates, Inc. (2012).

[2] Deng, J., Dong, W., Socher, R., Li, L.-J., Kai Li, Li Fei-Fei: ImageNet: A large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255. IEEE, Miami, FL (2009). https://doi.org/10.1109/CVPR.2009.5206848

[3] He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition. arXiv:1512.03385 [cs] (2015).

[4] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., Hassabis, D.: Human-level control through deep reinforcement learning. Nature 518(7540), 529–533 (2015). https://doi.org/10.1038/nature14236

[5] Walker, O., Vanegas, F., Gonzalez, F., Koenig, S.: A deep reinforcement learning framework for uav navigation in indoor environments. In: IEEE Aerospace Conference, pp. 1–14 (2019). https://doi.org/10.1109/AERO.2019.8742226

[6] Mirowski, P., Pascanu, R., Viola, F., Soyer, H., Ballard, A.J., Banino, A., Denil, M., Goroshin, R., Sifre, L., Kavukcuoglu, K., Kumaran, D., Hadsell, R.: Learning to Navigate in Complex Environments. arXiv:1611.03673 [cs] (2017).

[7] Cimurs, R., Suh, I.H., Lee, J.H.: Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning. arXiv:2103.07119 [cs] (2021).

[8] Wisniewski, M., Ona, I.G., Chatzithanos, P., Guo, W., Tsourdos, A.: Autonomous Navigation in Dynamic Maze Environments Under Adversarial Sensor Attack. In: 2024 International Conference on Unmanned Aircraft Systems (ICUAS), pp. 881–886 (2024). https://doi.org/10.1109/ICUAS60882.2024.10557088

[9] Zhu, K., Zhang, T.: Deep reinforcement learning based mobile robot navigation: A review. Tsinghua Science and Technology 26(5), 674–691 (2021). https://doi.org/10.26599/TST.2021.9010012

[10] Beattie, C., Leibo, J.Z., Teplyashin, D., Ward, T., Wainwright, M., Küttler, H., Lefrancq, A., Green, S., Valdés, V., Sadik, A., Schrittwieser, J., Anderson, K., York, S., Cant, M., Cain, A., Bolton, A., Gaffney, S., King, H., Hassabis, D., Legg, S., Petersen, S.: DeepMind Lab. arXiv:1612.03801 [cs] (2016).

[11] Kempka, M., Wydmuch, M., Runc, G., Toczek, J., Jaśkowski, W.: ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning. arXiv:1605.02097 [cs] (2016).

[12] Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T.P., Harley, T., Silver, D., Kavukcuoglu, K.: Asynchronous Methods for Deep Reinforcement Learning. arXiv:1602.01783 [cs] (2016).

[13] Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., Scaramuzza, D.: Champion-level drone racing using deep reinforcement learning. Nature 620(7976), 982–987 (2023). https://doi.org/10.1038/s41586-023-06419-4

[14] Polvara, R., Patacchiola, M., Sharma, S., Wan, J., Manning, A., Sutton, R., Cangelosi, A.: Toward end-to-end control for uav autonomous landing via deep reinforcement learning. In: International Conference on Unmanned Aircraft Systems (ICUAS) (2018).

[15] Chen, J., Li, S.E., Tomizuka, M.: Interpretable end-to-end urban autonomous driving with latent deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems 23(6) (2022).

[16] Pasukonis, J., Lillicrap, T., Hafner, D.: Evaluating Long-Term Memory in 3D Mazes. arXiv:2210.13383 [cs] (2022).

[17] Lample, G., Chaplot, D.S.: Playing FPS Games with Deep Reinforcement Learning. arXiv:1609.05521 [cs] (2018).

[18] Luo, C., Yang, S.X.: A bioinspired neural network for real-time concurrent map building and complete coverage robot navigation in unknown environments. IEEE Transactions on Neural Networks 19(7), 1279–1298 (2008). https://doi.org/10.1109/TNN.2008.2000394

[19] al., M.C.: Robust flight navigation out of distribution with liquid neural networks. Science Robotics 8 (2023).

[20] Sajjadi, S., Bittick, J., Janabi-Sharifi, F., Mantegh, I.: A robust and adaptive sensor fusion approach for indoor uav localization. In: International Conference on Unmanned Aircraft Systems (ICUAS) (2023).

[21] Negru, S.A., Geragersian, P., Petrunin, I., Guo, W.: Resilient multi-sensor uav navigation with a hybrid federated fusion architecture. Sensors 24 (2023).

[22] Liu, C., Zhang, G., Guo, W., He, R.: Kalman prediction-based neighbor discovery and its effect on routing protocol in vehicular ad hoc networks. IEEE Transactions on Intelligent Transportation Systems 21(1), 159–169 (2020). https://doi.org/10.1109/TITS.2018.2889923

[23] Bae, S., Han, D., Park, S.: A new approach to lidar and camera fusion for autonomous driving. In: IEEE International Conference on Artificial Intelligence in Information and Communication (2023).

[24] Kacker, T., Perrusquia, A., Guo, W.: Multi-spectral fusion using generative adversarial networks for uav detection of wild fires. In: IEEE Int. Conference on AI in Information and Communication (2023).

[25] Spaan, M.T.J.: Partially Observable Markov Decision Processes. In: Wiering, M., Otterlo, M. (eds.) Reinforcement Learning: State-of-the-Art, pp. 387–414. Springer, Berlin, Heidelberg (2012). https://doi.org/10.1007/978-3-642-27645-3_12

[26] Osogami, T.: Robust partially observable Markov decision process. In: Proceedings of the 32nd International Conference on Machine Learning, pp. 106–115. PMLR (2015).

[27] Tessler, C., Efroni, Y., Mannor, S.: Action Robust Reinforcement Learning and Applications in Continuous Control. In: Proceedings of the 36th International Conference on Machine Learning, pp. 6215–6224. PMLR (2019).

[28] Zhang, H., Chen, H., Xiao, C., Li, B., Liu, M., Boning, D., Hsieh, C.-J.: Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations. In: Advances in Neural Information Processing Systems, vol. 33, pp. 21024–21037. Curran Associates, Inc. (2020).

[29] Pinto, L., Davidson, J., Sukthankar, R., Gupta, A.: Robust Adversarial Reinforcement Learning. arXiv:1703.02702 [cs] (2017).

[30] Korkmaz, E.: Adversarial Training Blocks Generalization in Neural Policies. International Conference on Learning Representation (ICLR) Robust and Reliable Machine Learning in the Real World Workshop (2021).

[31] Korkmaz, E.: Adversarial Robust Deep Reinforcement Learning Requires Redefining Robustness. Proceedings of the AAAI Conference on Artificial Intelligence 37(7), 8369–8377 (2023). https://doi.org/10.1609/aaai.v37i7.26009

[32] Korkmaz, E.: Understanding and Diagnosing Deep Reinforcement Learning. arXiv:2406.16979 [cs] (2024).

[33] Havens, A.J., Jiang, Z., Sarkar, S.: Online Robust Policy Learning in the Presence of Unknown Adversaries. arXiv:1807.06064 [cs, stat] (2018).

[34] Panda, D.K., Guo, W.: Action Robust Reinforcement Learning for Air Mobility Deconfliction Against Conflict Induced Spoofing. IEEE Transactions on Intelligent Transportation Systems, 1–13 (2024). https://doi.org/10.1109/TITS.2024.3454354

[35] Towers, M., Kwiatkowski, A., Terry, J., Balis, J.U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J.J., Tan, H., Younis, O.G.: Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032 [cs] (2024).

[36] Kim, S.-G., Lee, E., Hong, I.-P., Yook, J.-G.: Review of Intentional Electromagnetic Interference on UAV Sensor Modules and Experimental Study. Sensors 22(6), 2384 (2022). https://doi.org/10.3390/s22062384

[37] Fujimoto, S., Hoof, H., Meger, D.: Addressing Function Approximation Error in Actor-Critic Methods. arXiv:1802.09477 [cs, stat] (2018).

[38] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs] (2017).

[39] Hochreiter, S., Schmidhuber, J.: Long Short-Term Memory. Neural Computation 9(8), 1735–1780 (1997). https://doi.org/10.1162/neco.1997.9.8.1735

[40] Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T.: Mastering Diverse Domains through World Models. arXiv:2301.04104 [cs, stat] (2023).

[41] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N.: Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research 22(268), 1–8 (2021).
