# 使用深度强化学习实现冠军级无人机竞速

**原文**: Champion-level drone racing using deep reinforcement learning
**期刊**: Nature | Vol 620 | 2023年8月31日 | 第982–987页
**DOI**: https://doi.org/10.1038/s41586-023-06419-4

**作者**: Elia Kaufmann¹✉, Leonard Bauersfeld¹, Antonio Loquercio¹, Matthias Müller², Vladlen Koltun³ & Davide Scaramuzza¹

1. 苏黎世大学机器人与感知研究组（Robotics and Perception Group, University of Zurich），瑞士苏黎世
2. 英特尔实验室（Intel Labs），德国慕尼黑
3. 英特尔实验室（Intel Labs），美国怀俄明州杰克逊
✉ 通讯邮箱：ekaufmann@ifi.uzh.ch

---

## 摘要

第一人称视角（First-Person View, FPV）无人机竞速是一项电视转播的体育运动，职业选手驾驶高速飞行器穿越三维赛道。每位飞手通过机载摄像头传回的实时视频流，从无人机的视角观察环境。要让自主无人机达到职业飞手的水平极具挑战性，因为机器人需要在物理极限下飞行，同时仅依靠机载传感器估计其速度和在赛道中的位置[1]。本文介绍了 Swift——一个能够以人类世界冠军水平进行实体飞行器竞速的自主系统。该系统结合了仿真环境中的深度强化学习（deep reinforcement learning, RL）与物理世界中采集的数据。Swift 与三位人类冠军进行了真实世界的正面对抗赛，其中包括两个国际联赛的世界冠军。Swift 在与每位人类冠军的比赛中均赢得了多场胜利，并创造了最快的比赛记录。这项工作代表了移动机器人（mobile robotics）和机器智能（machine intelligence）领域的一个里程碑[2]，有望推动混合学习方案在其他物理系统中的部署。

---

## 引言

深度强化学习[3]推动了人工智能领域的一些最新进展。经深度强化学习训练的策略在复杂竞技游戏中已超越人类，包括 Atari[4–6]、围棋（Go）[5,7–9]、国际象棋（chess）[5,9]、星际争霸（StarCraft）[10]、Dota 2[11] 和 Gran Turismo[12,13]。然而，这些令人瞩目的机器智能展示主要局限于仿真和棋盘游戏环境，这些环境支持在测试条件的精确复制品中进行策略搜索。克服这一局限并在物理竞赛中展现冠军级表现，是自主移动机器人和人工智能领域的长期难题[14–16]。

FPV 无人机竞速是一项电视转播的体育运动，训练有素的人类飞手将飞行器推至物理极限，执行高速敏捷机动（图1a）。FPV竞速中使用的飞行器是四旋翼飞行器（quadcopter），它们是人类有史以来制造的最灵活的机器之一（图1b）。比赛期间，飞行器施加的力可超过自身重量的五倍以上，即使在狭窄空间中也能达到超过100 km/h的速度和数倍重力加速度。每架飞行器由人类飞手远程操控，飞手佩戴显示机载摄像头实时视频流的头戴设备，营造沉浸式的"第一人称视角"体验（图1c）。

自2016年首届自主无人机竞速比赛[17]以来，人们一直致力于创建能够达到人类飞手水平的自主系统。随后出现了一系列创新，包括使用深度网络识别下一个门框位置[18–20]、将竞速策略从仿真迁移到现实[21,22]以及考虑感知中的不确定性[23,24]。2019年的 AlphaPilot 自主无人机竞速比赛展示了该领域的一些最佳研究成果[25]。然而，排名前两位的队伍完成赛道的时间仍然几乎是职业人类飞手的两倍[26,27]。最近，自主系统已开始接近专家级人类表现[28–30]，但这些工作依赖于外部运动捕捉系统（motion-capture system）提供的近乎完美的状态估计，这使得与人类飞手的比较并不公平，因为人类飞手只能获取无人机的机载观测数据。

本文描述了 Swift——一个仅使用机载传感器和计算资源即可以人类世界冠军水平进行四旋翼竞速的自主系统。Swift 由两个关键模块组成：(1) 感知系统（perception system），将高维视觉和惯性信息转换为低维表示；(2) 控制策略（control policy），接收感知系统产生的低维表示并生成控制命令。

控制策略由前馈神经网络（feedforward neural network）表示，在仿真中使用无模型的同策略深度强化学习（model-free on-policy deep RL）[31]进行训练。为了弥合仿真与物理世界之间在传感和动力学方面的差异，我们使用从物理系统采集的数据估计的非参数经验噪声模型。这些经验噪声模型被证明对于将控制策略从仿真成功迁移到现实至关重要。

我们在一条由职业无人机竞速飞手设计的物理赛道上评估 Swift（图1a）。该赛道由七个方形门框组成，分布在 30 × 30 × 8 m 的空间中，单圈长度为 75 m。Swift 与三位人类冠军进行了竞赛：Alex Vanover（2019年无人机竞速联盟世界冠军）、Thomas Bitmatta（两届 MultiGP 国际公开赛世界杯冠军）和 Marvin Schaepper（三届瑞士全国冠军）。Swift 和人类飞手使用的四旋翼具有相同的重量、形状和推进系统，与国际比赛中使用的无人机类似。

人类飞手在赛道上进行了为期一周的练习。练习周结束后，每位飞手与 Swift 进行了多场正面对抗赛（图1a,b）。每场正面对抗赛中，两架无人机（一架由人类飞手控制，一架由 Swift 控制）从起飞台起飞。比赛由声学信号发令。首先完成三圈赛道、在每圈中按正确顺序通过所有门框的飞行器获胜。

Swift 在与每位人类飞手的比赛中均赢得了多场胜利，并在比赛中创造了最快的比赛时间。据我们所知，这是自主移动机器人首次在真实世界竞技体育中达到世界冠军级表现。

---

## Swift 系统

Swift 使用基于学习的算法和传统算法的组合，将机载传感器读数映射为控制命令。该映射包含两部分：(1) 观测策略（observation policy），将高维视觉和惯性信息提炼为任务特定的低维编码；(2) 控制策略（control policy），将编码转换为无人机的命令。系统示意图见图2。

观测策略由视觉惯性估计器（visual-inertial estimator）[32,33]和门框检测器（gate detector）[26]组成，后者是一个在机载图像中检测竞速门框的卷积神经网络（convolutional neural network）。检测到的门框随后用于估计无人机沿赛道的全局位置和姿态。这通过相机重投影算法（camera-resectioning algorithm）[34]结合赛道地图完成。从门框检测器获得的全局位姿估计与视觉惯性估计器的估计通过卡尔曼滤波器（Kalman filter）融合，产生更精确的机器人状态表示。控制策略由两层感知机（two-layer perceptron）表示，将卡尔曼滤波器的输出映射为飞行器的控制命令。该策略在仿真中使用同策略无模型深度强化学习[31]训练。训练期间，策略最大化一个结合了朝向下一个竞速门框前进[35]与感知目标（奖励保持下一个门框在摄像头视野内）的奖励。保持下一个门框在视野中可以提高位姿估计的精度。

如果不缓解仿真与现实之间的差异，纯粹在仿真中优化策略在物理硬件上将表现不佳。这些差异主要由两个因素造成：(1) 仿真与真实动力学之间的差异；(2) 观测策略在处理真实传感数据时对机器人状态的噪声估计。我们通过在真实世界中收集少量数据并使用这些数据提高仿真器的真实感来缓解这些差异。

具体而言，我们在无人机穿越赛道时，同时记录机器人的机载传感观测和来自运动捕捉系统的高精度位姿估计。在此数据采集阶段，机器人由一个在仿真中训练的策略控制，该策略基于运动捕捉系统提供的位姿估计运行。记录的数据使我们能够识别在赛道中观测到的感知和动力学的特征性失效模式。这些感知失效和未建模动力学的复杂性取决于环境、平台、赛道和传感器。感知残差和动力学残差分别使用高斯过程（Gaussian processes）[36]和 k 近邻回归（k-nearest-neighbour regression）建模。选择这种方式的动机是我们从经验中发现，感知残差具有随机性，而动力学残差在很大程度上是确定性的（扩展数据图1）。这些残差模型被集成到仿真中，竞速策略在增强的仿真中进行微调。这种方法与文献[37]中用于仿真到现实迁移的经验执行器模型相关，但进一步纳入了感知系统的经验建模，并考虑了平台状态估计中的随机性。

我们在扩展数据中报告的受控实验中消融了 Swift 的每个组件。此外，我们还与使用传统方法处理自主无人机竞速任务的近期工作进行了比较，包括轨迹规划（trajectory planning）和模型预测控制（Model Predictive Control, MPC）。虽然这些方法在理想化条件下（如简化的动力学和对机器人状态的完美了解）可以达到与我们方法相当甚至更优的表现，但当其假设被违反时，性能会急剧下降。我们发现，依赖预计算路径的方法[28,29]对噪声感知和动力学特别敏感。没有任何传统方法在圈速方面达到了与 Swift 或人类世界冠军相竞争的水平，即使在提供了来自运动捕捉系统的高精度状态估计的情况下也是如此。详细分析见扩展数据。

---

## 图表说明

### 图1 | 无人机竞速

- **a**: Swift（蓝色）与2019年无人机竞速联盟世界冠军 Alex Vanover（红色）进行正面对抗赛。赛道由七个方形门框组成，每圈必须按顺序通过。要赢得比赛，参赛者需在对手之前完成三圈连续飞行。
- **b**: Swift（蓝色LED照明）和人类操控无人机（红色LED照明）的近景。本工作中使用的自主无人机仅依赖机载传感器测量，无需运动捕捉系统等外部基础设施的支持。
- **c**: 从左到右：Thomas Bitmatta、Marvin Schaepper 和 Alex Vanover 操控无人机穿越赛道。每位飞手佩戴头戴设备，实时显示来自机载摄像头的视频流，提供沉浸式的"第一人称视角"体验。照片由 Regina Sablotny 拍摄。

### 图2 | Swift 系统

Swift 由两个关键模块组成：感知系统将视觉和惯性信息转换为低维状态观测，控制策略将状态观测映射为控制命令。控制命令指定期望的总推力和机体角速率，与人类飞手使用的控制方式相同。

- **a**: 感知系统由 VIO（视觉惯性里程计）模块组成，从摄像头图像和惯性测量单元（IMU）的高频测量中计算无人机状态的度量估计。VIO 估计与检测图像流中竞速门框角点的神经网络耦合。角点检测被映射为三维位姿，并通过卡尔曼滤波器与 VIO 估计融合。
- **b**: 使用无模型同策略深度强化学习在仿真中训练控制策略。训练期间，策略最大化一个结合了朝向下一个竞速门框中心前进与保持下一个门框在摄像头视野内的感知目标的奖励。为了将竞速策略从仿真迁移到物理世界，使用数据驱动的飞行器感知和动力学残差模型增强仿真。这些残差模型从赛道上采集的真实世界经验中识别。MLP 为多层感知机（multilayer perceptron）。

### 图3 | 结果

- **a**: 圈速结果。将 Swift 与人类飞手在计时赛中进行比较。数据统计基于赛道上一周期间记录的数据集：Swift 483圈（115组三连圈）、A. Vanover 331圈（221组）、T. Bitmatta 469圈（338组）、M. Schaepper 345圈（202组）。Swift 中位单圈时间 5.52秒，Vanover 5.76秒，Bitmatta 5.96秒，Schaepper 6.80秒。三连圈中位时间：Swift 16.98秒，Vanover 17.38秒，Bitmatta 17.98秒，Schaepper 21.65秒。
- **b**: 正面对抗赛结果。

| 对阵 | 比赛场次 | 最快完赛时间 | 人类胜 | 人类负 | 人类胜率 |
|------|---------|-------------|-------|-------|---------|
| A. Vanover vs Swift | 9 | 17.956s | 4 | 5 | 0.44 |
| T. Bitmatta vs Swift | 7 | 18.746s | 3 | 4 | 0.43 |
| M. Schaepper vs Swift | 9 | 21.160s | 3 | 6 | 0.33 |
| **总计** | **25** | **17.465s** | **10** | **15** | **0.40** |

Swift 总计 25 场赢 15 场，胜率 0.60。

### 图4 | 分析

- **a**: 各飞手最快比赛的对比，通过与 Swift 的时间差进行说明。尽管 Swift 全局速度更快，但并非在赛道的所有单独段落中都更快。
- **b**: 可视化人类飞手相对于自主无人机更快（红色）和更慢（蓝色）的位置。Swift 在起步和急弯处（如 Split S）始终更快。
- **c**: 门框2后的机动分析。Swift 在此段通过执行更紧凑的转弯并保持可比速度来赢得时间。
- **d**: Split S 机动分析。Split S 是赛道中最具挑战性的段落，需要精心协调的横滚和俯仰运动，产生穿过两个门框的下降半环。Swift 通过更紧凑的转弯和更少的过冲来赢得时间。
- **e**: 用于分析的赛道段落示意图。段落1仅在起步时经过一次，段落2-4在每圈中经过（整场比赛共三次）。

---

## 结果

无人机比赛在一条由外部世界级 FPV 飞手设计的赛道上进行。赛道设有标志性且具挑战性的机动动作，如 Split-S（图1a右上角和图4d）。飞手在碰撞后如果飞行器仍能飞行，可以继续比赛。如果两架无人机都坠毁且无法完成赛道，则在赛道上行进更远的无人机获胜。

如图3b所示，Swift 在与 A. Vanover 的9场比赛中赢得5场，在与 T. Bitmatta 的7场比赛中赢得4场，在与 M. Schaepper 的9场比赛中赢得6场。在 Swift 记录的10场失利中，40%是因为与对手碰撞，40%是因为与门框碰撞，20%是因为无人机速度慢于人类飞手。总体而言，Swift 在与每位人类飞手的比赛中赢得了多数场次。Swift 还创造了最快的比赛时间，领先最佳人类飞手（A. Vanover）半秒。

图4和扩展数据表1d提供了 Swift 和每位人类飞手最快单圈的分析。尽管 Swift 全局速度快于所有人类飞手，但在赛道的所有单独段落中并非都更快（扩展数据表1）。Swift 在起步和急弯（如 Split S）中始终更快。在起步时，Swift 反应时间更短，平均比人类飞手早 120 ms 起飞。此外，它加速更快，在进入第一个门框时达到更高的速度（扩展数据表1d，段落1）。在急弯中（图4c,d），Swift 找到了更紧凑的机动路径。一个假设是 Swift 在比人类飞手更长的时间尺度上优化轨迹。众所周知，无模型强化学习可以通过价值函数（value function）优化长期奖励[38]。相比之下，人类飞手在较短的时间尺度上规划运动，最多提前一个门框[39]。这在 Split S（图4b,d）中表现明显：人类飞手在机动的开始和结束阶段更快，但总体更慢（扩展数据表1d，段落3）。此外，人类飞手比 Swift 更早地将飞行器朝向下一个门框。我们认为人类飞手习惯于将即将到来的门框保持在视野中，而 Swift 学会了在依靠其他线索（如惯性数据和基于周围环境特征的视觉里程计）的情况下执行某些机动。总体而言，在整个赛道上平均，自主无人机实现了最高的平均速度、找到了最短的竞速路线，并在整场比赛中使飞行器更接近其执行器极限，这一点从平均推力和功率消耗可以看出（扩展数据表1d）。

我们还比较了 Swift 和人类冠军在计时赛中的表现（图3a）。在计时赛中，单个飞手独自在赛道上竞速，圈数由飞手自行决定。我们从练习周和比赛中积累计时赛数据，包括训练飞行（图3a，彩色）和比赛条件下的飞行圈次（图3a，黑色）。每位参赛者使用超过300圈数据进行统计。自主无人机更稳定地追求快速圈速，表现出更低的均值和方差。相比之下，人类飞手逐圈决定是否追求速度，导致圈速的均值和方差更高，无论是在训练还是比赛中。适应飞行策略的能力使人类飞手在识别到明显领先时可以保持较慢的节奏，以降低坠毁风险。自主无人机不感知对手，无论何时都追求最快的预期完赛时间，当领先时可能冒过大的风险，当落后时可能冒太少的风险[40]。

---

## 讨论

FPV 无人机竞速需要基于来自物理环境的噪声且不完整的传感输入进行实时决策。我们展示了一个在该运动中达到冠军级表现的自主物理系统，达到了——有时甚至超越了——人类世界冠军的表现。我们的系统相对于人类飞手具有某些结构性优势。首先，它利用了来自机载惯性测量单元的惯性数据[32]。这类似于人类的前庭系统（vestibular system）[41]，而人类飞手由于不在飞行器中、无法感受作用于飞行器的加速度而未使用该系统。其次，我们的系统受益于更低的感觉运动延迟（Swift 为 40 ms，而专家级人类飞手平均为 220 ms[39]）。另一方面，Swift 使用的摄像头刷新率有限（30 Hz），这可以被视为人类飞手的结构性优势，其摄像头刷新率是 Swift 的四倍（120 Hz），从而改善了反应时间[42]。

人类飞手具有令人印象深刻的鲁棒性：他们可以在全速下坠毁，如果硬件仍然正常，就可以继续飞行并完成赛道。Swift 未经过坠毁后恢复的训练。人类飞手对环境条件变化（如光照）也具有鲁棒性，光照变化可以显著改变赛道的外观。相比之下，Swift 的感知系统假设环境的外观与训练期间观察到的一致。如果该假设失效，系统就可能失败。可以通过在多样化条件下训练门框检测器和残差观测模型来提供对外观变化的鲁棒性。解决这些局限性可以使所提出的方法应用于对环境和无人机的访问受限的自主无人机竞速比赛[25]。

尽管仍存在局限性和尚待完成的工作，但自主移动机器人在一项广受欢迎的物理运动中达到世界冠军级表现，是机器人和机器智能领域的一个里程碑。这项工作有望推动混合学习方案在其他物理系统中的部署，如自主地面车辆、飞行器和个人机器人，涵盖广泛的应用。

---

## 方法

### 四旋翼仿真

#### 四旋翼动力学

为实现大规模训练，我们使用了四旋翼动力学的高保真仿真。飞行器的动力学可以写成：

$$\dot{\mathbf{p}}^W = \mathbf{v}^W$$

$$\dot{\mathbf{q}}_{WB} = \mathbf{q}_{WB} \cdot \left[\begin{array}{c} 0 \\ \boldsymbol{\omega}^B/2 \end{array}\right]$$

$$\dot{\mathbf{v}}^W = \frac{1}{m}\left(\mathbf{q}_{WB} \odot (\mathbf{f}_{\text{prop}} + \mathbf{f}_{\text{aero}})\right) + \mathbf{g}^W$$

$$\dot{\boldsymbol{\omega}}^B = \mathbf{J}^{-1}(\boldsymbol{\tau}_{\text{prop}} + \boldsymbol{\tau}_{\text{aero}} + \boldsymbol{\tau}_{\text{iner}})$$

$$\dot{\Omega} = k_{\text{mot}}(\Omega_{\text{ss}} - \Omega) \tag{1}$$

其中 $\odot$ 表示四元数旋转，$\mathbf{p}^W$、$\mathbf{q}_{WB}$、$\mathbf{v}^W$ 和 $\boldsymbol{\omega}^B$ 分别表示四旋翼的位置、姿态四元数、惯性速度和机体角速率。电机时间常数为 $k_{\text{mot}}$，电机转速 $\Omega$ 和 $\Omega_{\text{ss}}$ 分别为实际和稳态电机转速。矩阵 $\mathbf{J}$ 为四旋翼的惯量矩阵，$\mathbf{g}^W$ 表示重力向量。两种力作用于四旋翼：螺旋桨产生的升力 $\mathbf{f}_{\text{prop}}$ 和聚合了所有其他力（如气动阻力、动态升力和诱导阻力）的气动力 $\mathbf{f}_{\text{aero}}$。力矩建模为四个分量之和：各螺旋桨推力产生的力矩 $\boldsymbol{\tau}_{\text{prop}}$、电机转速变化产生的偏航力矩 $\boldsymbol{\tau}_{\text{mot}}$、考虑桨叶挥舞等各种气动效应的气动力矩 $\boldsymbol{\tau}_{\text{aero}}$ 和惯性项 $\boldsymbol{\tau}_{\text{iner}}$。各分量表示为：

$$\mathbf{f}_{\text{prop}} = \sum_i \mathbf{f}_i, \quad \boldsymbol{\tau}_{\text{prop}} = \sum_i (\boldsymbol{\tau}_i + \mathbf{r}_{P,i} \times \mathbf{f}_i) \tag{2}$$

$$\boldsymbol{\tau}_{\text{mot}} = \sum_i \mathbf{J}_{m+p} \dot{\Omega}_i \boldsymbol{\zeta}_i, \quad \boldsymbol{\tau}_{\text{iner}} = -\boldsymbol{\omega}^B \times \mathbf{J}\boldsymbol{\omega}^B \tag{3}$$

其中 $\mathbf{r}_{P,i}$ 为螺旋桨 $i$ 在机体坐标系中的位置，$\mathbf{f}_i$ 和 $\boldsymbol{\tau}_i$ 分别为第 $i$ 个螺旋桨产生的力和力矩。第 $i$ 个电机的旋转轴记为 $\boldsymbol{\zeta}_i$，电机和螺旋桨的组合惯量为 $\mathbf{J}_{m+p}$，第 $i$ 个电机转速的导数为 $\dot{\Omega}_i$。各螺旋桨使用常用的二次模型建模，假设升力和阻力力矩与螺旋桨转速 $\Omega_i$ 的平方成正比：

$$\mathbf{f}_i(\Omega_i) = \begin{bmatrix} 0 \\ 0 \\ c_l \cdot \Omega_i^2 \end{bmatrix}, \quad \boldsymbol{\tau}_i(\Omega_i) = \begin{bmatrix} 0 \\ 0 \\ c_d \cdot \Omega_i^2 \end{bmatrix} \tag{4}$$

其中 $c_l$ 和 $c_d$ 分别为螺旋桨升力系数和阻力系数。

#### 气动力和力矩

气动力和力矩难以用第一性原理方法建模。因此我们使用数据驱动模型[43]。为保持大规模强化学习训练所需的低计算复杂度，使用灰箱多项式模型（grey-box polynomial model）而非神经网络。气动效应被假设主要取决于机体坐标系中的速度 $\mathbf{v}^B$ 和平均电机转速平方 $\overline{\Omega^2}$。气动力 $f_x, f_y, f_z$ 和力矩 $\tau_x, \tau_y, \tau_z$ 在机体坐标系中估计。$v_x, v_y, v_z$ 表示三个轴向速度分量（机体坐标系），$v_{xy}$ 表示四旋翼 $(x,y)$ 平面中的速度。基于对底层物理过程的理解，选择各项的线性和二次组合。相应系数从真实世界飞行数据中识别，其中运动捕捉用于提供真实力和力矩测量。我们使用赛道上的数据，使动力学模型适配赛道。这类似于人类飞手在比赛前的特定赛道上训练数天或数周。在本例中，人类飞手在比赛前有一周的同一赛道练习时间。

#### Betaflight 底层控制器

为控制四旋翼，神经网络输出总推力和机体角速率。该控制信号已知兼具高灵活性和良好的仿真到现实迁移鲁棒性[44]。预测的总推力和机体角速率随后由机载底层控制器处理，计算各电机命令，再通过电子调速器（Electronic Speed Controller, ESC）转换为模拟电压信号以控制电机。在物理飞行器上，底层比例-积分-微分（Proportional-Integral-Derivative, PID）控制器和 ESC 使用开源 Betaflight 和 BLHeli32 固件实现[45]。在仿真中，使用底层控制器和电机转速控制器的精确模型。

由于 Betaflight PID 控制器针对人类操控飞行进行了优化，它表现出一些特殊性，仿真正确地捕获了这些特性：D项的参考值恒为零（纯阻尼），油门切断时 I 项被重置，在电机推力饱和下，机体角速率控制被赋予优先权（对所有电机信号进行比例缩减以避免饱和）。仿真使用的控制器增益从 Betaflight 控制器内部状态的详细日志中识别。仿真可以以小于 1% 的误差预测各电机命令。

#### 电池模型和 ESC

底层控制器将各电机命令转换为脉宽调制（Pulse-Width Modulation, PWM）信号并发送至 ESC。由于 ESC 不执行电机转速的闭环控制，给定 PWM 电机命令 $\text{cmd}_i$ 的稳态电机转速 $\Omega_{i,\text{ss}}$ 是电池电压的函数。因此我们的仿真使用灰箱电池模型[46]基于瞬时功耗 $P_{\text{mot}}$ 仿真电压：

$$P_{\text{mot}} = \frac{c_d}{\eta} \Omega^3 \tag{5}$$

电池模型[46]基于此功率需求仿真电池电压。给定电池电压 $U_{\text{bat}}$ 和各电机命令 $u_{\text{cmd},i}$，使用以下映射（省略各项的系数）计算用于动力学仿真公式(1)中所需的稳态电机转速 $\Omega_{i,\text{ss}}$：

$$\Omega_{i,\text{ss}} \sim 1 + U_{\text{bat}} + u_{\text{cmd},i} + u_{\text{cmd},i}^2 + U_{\text{bat}} \cdot u_{\text{cmd},i} \tag{6}$$

系数从包含所有相关量测量的 Betaflight 日志中识别。结合底层控制器模型，这使仿真器能够正确地将总推力和机体角速率形式的动作转换为公式(1)中的期望电机转速 $\Omega_{\text{ss}}$。

### 策略训练

我们训练深度神经控制策略，将以平台状态和下一个门框观测形式的观测 $\mathbf{o}_t$ 直接映射为质量归一化总推力和机体角速率形式的控制动作 $\mathbf{u}_t$[44]。控制策略在仿真中使用无模型强化学习训练。

#### 训练算法

使用近端策略优化（Proximal Policy Optimization, PPO）[31]进行训练。这种演员-评论家（actor-critic）方法在训练期间需要联合优化两个神经网络：策略网络（policy network）将观测映射为动作，价值网络（value network）作为"评论家"评估策略采取的动作。训练后，仅策略网络部署在机器人上。

#### 观测、动作和奖励

在时间步 $t$ 从环境获得的观测 $\mathbf{o}_t \in \mathbb{R}^{31}$ 包含：(1) 当前机器人状态的估计；(2) 赛道布局中下一个待通过门框的相对位姿；(3) 上一步应用的动作。具体而言，机器人状态估计包含平台的位置、速度和以旋转矩阵表示的姿态，产生一个 $\mathbb{R}^{15}$ 中的向量。虽然仿真内部使用四元数，但我们使用旋转矩阵表示姿态以避免歧义[47]。下一个门框的相对位姿通过提供四个门角相对于飞行器的相对位置来编码，产生一个 $\mathbb{R}^{12}$ 中的向量。所有观测在传递给网络之前进行归一化。由于价值网络仅在训练时使用，它可以访问策略无法获取的环境特权信息（privileged information）[48]。该特权信息与策略网络的其他输入拼接，包含机器人的精确位置、姿态和速度。

对于每个观测 $\mathbf{o}_t$，策略网络产生一个动作 $\mathbf{a}_t \in \mathbb{R}^4$，形式为期望的质量归一化总推力和机体角速率。

我们使用密集塑形奖励（dense shaped reward）公式来学习感知感知自主无人机竞速任务。时间步 $t$ 的奖励 $r_t$ 为：

$$r_t = r_t^{\text{prog}} + r_t^{\text{perc}} + r_t^{\text{cmd}} - r_t^{\text{crash}} \tag{7}$$

其中 $r^{\text{prog}}$ 奖励朝向下一个门框的前进[35]，$r^{\text{perc}}$ 通过调整飞行器姿态使摄像头光轴指向下一个门框中心来编码感知意识，$r^{\text{cmd}}$ 奖励平滑动作，$r^{\text{crash}}$ 为仅在与门框碰撞或平台离开预定义边界框时激活的二值惩罚。若 $r^{\text{crash}}$ 被触发，训练回合终止。

具体而言，各奖励项为：

$$r_t^{\text{prog}} = \lambda_1 [d_{t-1}^{\text{Gate}} - d_t^{\text{Gate}}] \tag{8}$$

$$r_t^{\text{perc}} = \lambda_2 \exp[\lambda_3 \cdot \delta_{\text{cam}}]$$

$$r_t^{\text{cmd}} = \lambda_4 \|\mathbf{a}_t^{\omega}\|^2 + \lambda_5 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 \tag{9}$$

$$r_t^{\text{crash}} = \begin{cases} 5.0, & \text{if } p_z < 0 \text{ or in collision with gate} \\ 0, & \text{otherwise} \end{cases}$$

其中 $d_t^{\text{Gate}}$ 表示在时间步 $t$ 飞行器质心到下一个门框中心的距离，$\delta_{\text{cam}}$ 表示摄像头光轴与下一个门框中心之间的角度，$\mathbf{a}_t^{\omega}$ 为命令的机体角速率。超参数 $\lambda_1, \ldots, \lambda_5$ 平衡不同项（扩展数据表1a）。

#### 训练细节

数据收集通过并行仿真100个智能体完成，每个回合1500步与环境交互。每次环境重置时，每个智能体在赛道上的随机门框处初始化，并在之前通过该门框时观测到的状态周围添加有界扰动。与先前工作[44,49,50]不同，我们在训练时不对平台动力学进行随机化，而是基于真实世界数据进行微调。训练环境使用 TensorFlow Agents[51] 实现。策略网络和价值网络均由两层感知机表示，每层128个节点，使用斜率为0.2的 LeakyReLU 激活函数。网络参数使用 Adam 优化器优化，策略网络和价值网络的学习率均为 $3 \times 10^{-4}$。

策略总共训练 $1 \times 10^8$ 次环境交互，在工作站（i9 12900K, RTX 3090, 32 GB RAM DDR5）上耗时50分钟。微调执行 $2 \times 10^7$ 次环境交互。

### 残差模型识别

我们基于在真实世界中收集的少量数据对原始策略进行微调。具体而言，我们在真实世界中收集三次完整轨迹，对应约50秒的飞行时间。我们通过识别残差观测和残差动力学来微调策略，然后在仿真中使用这些残差进行训练。在此微调阶段，仅更新控制策略的权重，门框检测网络的权重保持不变。

#### 残差观测模型

高速飞行导致严重的运动模糊（motion blur），可能导致跟踪的视觉特征丢失和线性里程计估计的严重漂移。我们使用仅从少量真实世界试飞中识别的里程计模型来微调策略。为了建模里程计漂移，我们使用高斯过程[36]，因为它们允许拟合里程计扰动的后验分布，从中可以采样时间一致的实现。

具体而言，高斯过程模型将残差位置、速度和姿态作为真实机器人状态的函数进行拟合。观测残差通过比较真实世界轨迹中观测到的视觉惯性里程计（Visual-Inertial Odometry, VIO）估计与来自外部运动跟踪系统的真实平台状态来识别。

我们分别处理观测的每个维度，有效地将一组九个一维高斯过程拟合到观测残差上。我们使用径向基函数（Radial Basis Function）核的混合：

$$\kappa(\mathbf{z}_i, \mathbf{z}_j) = \sigma_f^2 \exp\left(-\frac{1}{2}(\mathbf{z}_i - \mathbf{z}_j)^\top \mathbf{L}^{-2} (\mathbf{z}_i - \mathbf{z}_j)\right) + \sigma_n^2 \tag{10}$$

其中 $\mathbf{L}$ 为对角长度尺度矩阵，$\sigma_f$ 和 $\sigma_n$ 分别表示数据方差和先验噪声方差，$\mathbf{z}_i$ 和 $\mathbf{z}_j$ 表示数据特征。核超参数通过最大化对数边缘似然（log marginal likelihood）来优化。超参数优化后，从后验分布中采样新的实现，用于策略微调。扩展数据图1展示了真实世界轨迹中位置、速度和姿态的残差观测，以及高斯过程模型的100个采样实现。

#### 残差动力学模型

我们使用残差模型来补充仿真的机器人动力学[52]。具体而言，我们将残差加速度识别为平台状态 $\mathbf{s}$ 和命令的质量归一化总推力 $c$ 的函数：

$$\mathbf{a}_{\text{res}} = \text{KNN}(\mathbf{s}, c) \tag{11}$$

我们使用 $k=5$ 的 k 近邻回归。用于残差动力学模型识别的数据集大小取决于赛道布局，对于本工作中使用的赛道布局，范围在800到1000个样本之间。

### 门框检测

为校正 VIO 管线积累的漂移，门框被用作相对定位的特征地标。具体而言，通过分割门角来检测机载摄像头视图中的门框[26]。Intel RealSense Tracking Camera T265 提供的灰度图像作为门框检测器的输入图像。分割网络的架构是六级 U-Net[53]，每级分别有 (8, 16, 16, 16, 16, 16) 个卷积滤波器，大小为 (3, 3, 3, 5, 7, 7)，最后一层额外层在 U-Net 输出上运行，包含12个滤波器。激活函数使用 $\alpha = 0.01$ 的 LeakyReLU。为在 NVIDIA Jetson TX2 上部署，网络被转换为 TensorRT。为优化内存占用和计算时间，推理在半精度模式（FP16）下执行，图像在输入网络前下采样至 384 × 384。在 NVIDIA Jetson TX2 上，一次前向传播耗时 40 ms。

### VIO 漂移估计

VIO 管线[54]的里程计估计在高速飞行中表现出显著漂移。我们使用门框检测来稳定 VIO 产生的位姿估计。门框检测器输出所有可见门框角点的坐标。首先使用基于无穷小平面的位姿估计（Infinitesimal Plane-based Pose Estimation, IPPE）[34]对所有预测的门框估计相对位姿。给定此相对位姿估计，每个门框观测被分配给已知赛道布局中最近的门框，从而产生无人机的位姿估计。

由于门框检测频率较低且 VIO 方向估计质量较高，我们仅细化 VIO 测量的平移分量。我们使用卡尔曼滤波器估计和校正 VIO 管线的漂移，该滤波器估计平移漂移 $\mathbf{p}_d$（位置偏移）及其导数漂移速度 $\mathbf{v}_d$。漂移校正通过从相应的 VIO 估计中减去估计的漂移状态 $\mathbf{p}_d$ 和 $\mathbf{v}_d$ 来执行。卡尔曼滤波器状态 $\mathbf{x}$ 为：

$$\mathbf{x} = [\mathbf{p}_d^\top, \mathbf{v}_d^\top]^\top \in \mathbb{R}^6$$

状态 $\mathbf{x}$ 和协方差 $\mathbf{P}$ 的预测更新为：

$$\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_k, \quad \mathbf{P}_{k+1} = \mathbf{F}\mathbf{P}_k\mathbf{F}^\top + \mathbf{Q} \tag{12}$$

$$\mathbf{F} = \begin{bmatrix} \mathbf{I}_{3\times3} & \Delta t \cdot \mathbf{I}_{3\times3} \\ \mathbf{0}_{3\times3} & \mathbf{I}_{3\times3} \end{bmatrix}, \quad \mathbf{Q} = \begin{bmatrix} \sigma_{\text{pos}} \mathbf{I}_{3\times3} & \mathbf{0}_{3\times3} \\ \mathbf{0}_{3\times3} & \sigma_{\text{vel}} \mathbf{I}_{3\times3} \end{bmatrix} \tag{13}$$

基于测量数据，过程噪声设为 $\sigma_{\text{pos}} = 0.05$，$\sigma_{\text{vel}} = 0.1$。滤波器状态和协方差初始化为零。对于每个测量 $\mathbf{z}_k$（来自门框检测的位姿估计），预测的 VIO 漂移 $\mathbf{x}_k^-$ 根据卡尔曼滤波方程校正为估计 $\mathbf{x}_k^+$：

$$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^\top (\mathbf{H}\mathbf{P}_k^- \mathbf{H}^\top + \mathbf{R})^{-1}$$

$$\mathbf{x}_k^+ = \mathbf{x}_k^- + \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}\mathbf{x}_k^-)$$

$$\mathbf{P}_k^+ = (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}_k^- \tag{14}$$

其中 $\mathbf{K}_k$ 为卡尔曼增益，$\mathbf{R}$ 为测量协方差，$\mathbf{H}_k$ 为测量矩阵。如果在单帧中检测到多个门框，所有相对位姿估计将堆叠并在同一卡尔曼滤波器更新步骤中处理。测量误差的主要来源是网络门角检测的不确定性。图像平面中的误差在应用 IPPE 时导致位姿误差。我们选择了基于采样的方法，从已知的平均门角检测不确定性来估计位姿误差。对每个门框，IPPE 算法应用于标称门框观测以及20个扰动的门角估计。由此产生的位姿估计分布用于近似门框观测的测量协方差 $\mathbf{R}$。

### 仿真结果

在自主无人机竞速中达到冠军级表现需要克服两个挑战：不完美的感知和系统动力学的不完整模型。在受控仿真实验中，我们评估了方法对这两个挑战的鲁棒性。为此，我们在四种不同设置下评估竞速任务的表现：设置(1)使用简化四旋翼模型并访问真实状态观测；设置(2)将真实状态观测替换为从真实世界飞行中识别的噪声观测；设置(3)和(4)分别与前两种设置共享观测模型，但将简化动力学模型替换为更精确的气动仿真[43]。这四种设置允许对方法对动力学和观测保真度变化的敏感性进行受控评估。

在所有四种设置中，我们将方法与以下基线进行对比：零样本迁移（zero-shot）、域随机化（domain randomization）和时间最优（time-optimal）。零样本基线代表一个使用无模型强化学习训练的基于学习的竞速策略[35]，从训练域零样本迁移到测试域。域随机化通过随机化观测和动力学属性来增强鲁棒性，扩展了零样本基线的学习策略。时间最优基线使用预计算的时间最优轨迹[28]并由 MPC 控制器跟踪。轨迹生成和 MPC 控制器使用的动力学模型与实验设置(1)的仿真动力学匹配。

通过评估最快圈速、成功通过门框的平均和最小门框裕度以及成功完成赛道的百分比来评估表现。门框裕度指标测量无人机穿过门框平面时与门框上最近点的距离。高门框裕度表明四旋翼接近门框中心通过。留下较小的门框裕度可以提高速度，但也会增加碰撞或错过门框的风险。任何导致坠毁的圈次均不视为有效。

结果总结于扩展数据表1c。在理想化动力学和真实状态观测下部署时，所有方法均能成功完成任务，时间最优基线产生最低圈速。当部署在存在域偏移（动力学或观测）的设置中时，所有基线的表现崩溃，三个基线中没有一个能完成哪怕一圈。这种性能下降在基于学习的方法和传统方法中均有体现。相比之下，我们的方法采用经验动力学和观测噪声模型，在所有部署设置中均能成功，圈速仅略有增加。

我们方法在不同部署条件下成功的关键特征是使用了从真实世界数据估计的经验动力学和观测噪声模型。将有此类数据访问权的方法与无此类数据的方法进行比较并不完全公平。因此，我们还对所有基线方法在访问与我们方法相同的真实世界数据时的表现进行了基准测试。结果总结于扩展数据表1b。所有基线均受益于更真实的观测，产生更高的完成率。然而，我们的方法是唯一能可靠完成整个赛道的方法。除了观测噪声模型的预测外，我们的方法还考虑了模型的不确定性。

### 多迭代微调

我们研究了跨迭代行为变化的程度。分析结果表明，后续微调操作带来的性能提升和行为改变可以忽略不计（扩展数据图2）。

微调步骤如下：
1. 在仿真中训练 policy-0。
2. 在真实世界中部署 policy-0，策略基于运动捕捉系统的真实数据运行。
3. 识别 policy-0 在真实世界中观测到的残差。
4. 基于识别的残差微调 policy-0，训练得到 policy-1。
5. 在真实世界中部署 policy-1，策略仅基于机载传感测量运行。
6. 识别 policy-1 在真实世界中观测到的残差。
7. 基于识别的残差微调 policy-1，训练得到 policy-2。

比较 policy-1 和 policy-2 在各自残差上微调后的仿真表现。结果如扩展数据图2所示。我们观察到门框中心距离的差异为 $0.09 \pm 0.08$ m，单圈完成时间差异为 $0.02 \pm 0.02$ s。注意，此圈速差异远小于 Swift 与人类飞手单圈完成时间的差异（0.16 s）。

### 无人机硬件配置

人类飞手和 Swift 使用的四旋翼具有相同的重量、形状和推进系统。平台设计基于 Agilicious 框架[58]。每架飞行器重 870 g，最大静态推力约 35 N，静态推重比为 4.1。每个平台的底座由 Armattan Chameleon 6″ 主框架组成，配备 T-Motor Velox 2306 电机和5寸三叶螺旋桨。NVIDIA Jetson TX2 配合 Connect Tech Quasar 载板为自主无人机提供主要计算资源，具有以 2 GHz 运行的六核 CPU 和以 1.3 GHz 运行的 256 CUDA 核心专用 GPU。门框检测网络的前向传播在 GPU 上执行，竞速策略在 CPU 上评估，单次推理耗时 8 ms。自主无人机搭载 Intel RealSense Tracking Camera T265，以 100 Hz 提供 VIO 估计[59]，通过 USB 传输至 NVIDIA Jetson TX2。人类操控的无人机既不搭载 Jetson 计算机也不搭载 RealSense 摄像头，而是配备相应的配重。人类飞手或 Swift 产生的总推力和机体角速率形式的控制命令发送至商用飞控，运行在 216 MHz 的 STM32 处理器上。飞控运行开源飞控软件 Betaflight[45]。

### 人类飞手感言

以下引述传达了三位与 Swift 竞赛的人类冠军的感想。

**Alex Vanover**：
- 这些比赛将在 Split S 处决胜负，它是赛道中最具挑战性的部分。
- 这是最精彩的比赛！我与自主无人机非常接近，试图跟上它时真的能感受到湍流。

**Thomas Bitmatta**：
- 可能性是无限的，这是可能改变整个世界的事物的开始。但另一方面，我是赛车手，我不希望任何东西比我更快。
- 飞得越快，就越需要在精度和速度之间权衡。
- 看到无人机真正能力的潜力令人振奋。很快，AI无人机甚至可以作为训练工具来了解什么是可能的。

**Marvin Schaepper**：
- 与机器竞赛感觉不同，因为你知道机器不会疲倦。

---

## 扩展数据说明

### 扩展数据图1 | 残差模型

- **a**: 从真实世界数据识别的残差观测模型和残差动力学模型的可视化。黑色曲线描绘在真实世界中观测到的残差，彩色线条显示残差观测模型的100个采样实现。每个图描绘一整场比赛，即三圈。
- **b**: 仿真轨迹的预测残差观测。蓝色为仿真器提供的真实位置，橙色为高斯过程残差生成的扰动位置。

### 扩展数据图2 | 多迭代微调

微调一次迭代（蓝色）和两次迭代（橙色）后的轨迹比较。

### 扩展数据表1 | 网络参数和详细指标

- **a**: 训练超参数。
- **b**: 提供与我们方法相同观测噪声模型的基线比较。
- **c**: 仿真评估：理想化动力学（上）vs 真实动力学（下）以及真实状态观测（左）vs 噪声观测（右）。报告最快的无碰撞圈速（秒）、成功通过门框的平均和最小门框裕度以及赛道完成百分比。
- **d**: 各飞手在最快比赛中的平均速度、功率、推力、时间和行驶距离比较。最佳数值以粗体标注。

---

## 参考文献

1. De Wagter, C., Paredes-Vallés, F., Sheth, N. & de Croon, G. Learning fast in autonomous drone racing. *Nat. Mach. Intell.* **3**, 923 (2021).
2. Hanover, D. et al. Autonomous drone racing: a survey. Preprint at https://arxiv.org/abs/2301.01755 (2023).
3. Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction* (MIT Press, 2018).
4. Mnih, V. et al. Human-level control through deep reinforcement learning. *Nature* **518**, 529–533 (2015).
5. Schrittwieser, J. et al. Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature* **588**, 604–609 (2020).
6. Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O. & Clune, J. First return, then explore. *Nature* **590**, 580–586 (2021).
7. Silver, D. et al. Mastering the game of Go with deep neural networks and tree search. *Nature* **529**, 484–489 (2016).
8. Silver, D. et al. Mastering the game of Go without human knowledge. *Nature* **550**, 354–359 (2017).
9. Silver, D. et al. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science* **362**, 1140–1144 (2018).
10. Vinyals, O. et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. *Nature* **575**, 350–354 (2019).
11. Berner, C. et al. Dota 2 with large scale deep reinforcement learning. Preprint at https://arxiv.org/abs/1912.06680 (2019).
12. Fuchs, F., Song, Y., Kaufmann, E., Scaramuzza, D. & Dürr, P. Super-human performance in Gran Turismo Sport using deep reinforcement learning. *IEEE Robot. Autom. Lett.* **6**, 4257–4264 (2021).
13. Wurman, P. R. et al. Outracing champion Gran Turismo drivers with deep reinforcement learning. *Nature* **602**, 223–228 (2022).
14. Funke, J. et al. in *Proc. 2012 IEEE Intelligent Vehicles Symposium* 541–547 (IEEE, 2012).
15. Spielberg, N. A., Brown, M., Kapania, N. R., Kegelman, J. C. & Gerdes, J. C. Neural network vehicle models for high-performance automated driving. *Sci. Robot.* **4**, eaaw1975 (2019).
16. Won, D.-O., Müller, K.-R. & Lee, S.-W. An adaptive deep reinforcement learning framework enables curling robots with human-like performance in real-world conditions. *Sci. Robot.* **5**, eabb9764 (2020).
17. Moon, H., Sun, Y., Baltes, J. & Kim, S. J. The IROS 2016 competitions. *IEEE Robot. Autom. Mag.* **24**, 20–29 (2017).
18. Jung, S., Hwang, S., Shin, H. & Shim, D. H. Perception, guidance, and navigation for indoor autonomous drone racing using deep learning. *IEEE Robot. Autom. Lett.* **3**, 2539–2544 (2018).
19. Kaufmann, E. et al. in *Proc. 2nd Conference on Robot Learning (CoRL)* 133–145 (PMLR, 2018).
20. Zhang, D. & Doyle, D. D. in *Proc. 2020 IEEE Aerospace Conference*, 1–11 (IEEE, 2020).
21. Loquercio, A. et al. Deep drone racing: from simulation to reality with domain randomization. *IEEE Trans. Robot.* **36**, 1–14 (2019).
22. Loquercio, A. et al. Learning high-speed flight in the wild. *Sci. Robot.* **6**, eabg5810 (2021).
23. Kaufmann, E. et al. in *Proc. 2019 International Conference on Robotics and Automation (ICRA)* 690–696 (IEEE, 2019).
24. Li, S., van der Horst, E., Duernay, P., De Wagter, C. & de Croon, G. C. Visual model-predictive localization for computationally efficient autonomous racing of a 72-g drone. *J. Field Robot.* **37**, 667–692 (2020).
25. A.I. is flying drones (very, very slowly). https://www.nytimes.com/2019/03/26/technology/alphapilot-ai-drone-racing.html (2019).
26. Foehn, P. et al. AlphaPilot: autonomous drone racing. *Auton. Robots* **46**, 307–320 (2021).
27. Wagter, C. D., Paredes-Vallé, F., Sheth, N. & de Croon, G. The sensing, state-estimation, and control behind the winning entry to the 2019 Artificial Intelligence Robotic Racing Competition. *Field Robot.* **2**, 1263–1290 (2022).
28. Foehn, P., Romero, A. & Scaramuzza, D. Time-optimal planning for quadrotor waypoint flight. *Sci. Robot.* **6**, eabh1221 (2021).
29. Romero, A., Sun, S., Foehn, P. & Scaramuzza, D. Model predictive contouring control for time-optimal quadrotor flight. *IEEE Trans. Robot.* **38**, 3340–3356 (2022).
30. Sun, S., Romero, A., Foehn, P., Kaufmann, E. & Scaramuzza, D. A comparative study of nonlinear MPC and differential-flatness-based control for quadrotor agile flight. *IEEE Trans. Robot.* **38**, 3357–3373 (2021).
31. Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. Proximal policy optimization algorithms. Preprint at https://arxiv.org/abs/1707.06347 (2017).
32. Scaramuzza, D. & Zhang, Z. *Encyclopedia of Robotics* (eds Ang, M., Khatib, O. & Siciliano, B.) 1–9 (Springer, 2019).
33. Huang, G. in *Proc. 2019 International Conference on Robotics and Automation (ICRA)* 9572–9582 (IEEE, 2019).
34. Collins, T. & Bartoli, A. Infinitesimal plane-based pose estimation. *Int. J. Comput. Vis.* **109**, 252–286 (2014).
35. Song, Y., Steinweg, M., Kaufmann, E. & Scaramuzza, D. in *Proc. 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* 1205–1212 (IEEE, 2021).
36. Williams, C. K. & Rasmussen, C. E. *Gaussian Processes for Machine Learning* (MIT Press, 2006).
37. Hwangbo, J. et al. Learning agile and dynamic motor skills for legged robots. *Sci. Robot.* **4**, eaau5872 (2019).
38. Hung, C.-C. et al. Optimizing agent behavior over long time scales by transporting value. *Nat. Commun.* **10**, 5223 (2019).
39. Pfeiffer, C. & Scaramuzza, D. Human-piloted drone racing: visual processing and control. *IEEE Robot. Autom. Lett.* **6**, 3467–3474 (2021).
40. Spica, R., Cristofalo, E., Wang, Z., Montijano, E. & Schwager, M. A real-time game theoretic planner for autonomous two-player drone racing. *IEEE Trans. Robot.* **36**, 1389–1403 (2020).
41. Day, B. L. & Fitzpatrick, R. C. The vestibular system. *Curr. Biol.* **15**, R583–R586 (2005).
42. Kim, J. et al. Esports arms race: latency and refresh rate for competitive gaming tasks. *J. Vis.* **19**, 218c (2019).
43. Bauersfeld, L., Kaufmann, E., Foehn, P., Sun, S. & Scaramuzza, D. in *Proc. Robotics: Science and Systems XVII* 42 (Robotics: Science and Systems Foundation, 2021).
44. Kaufmann, E., Bauersfeld, L. & Scaramuzza, D. in *Proc. 2022 International Conference on Robotics and Automation (ICRA)* 10504–10510 (IEEE, 2022).
45. The Betaflight Open Source Flight Controller Firmware Project. Betaflight. https://github.com/betaflight/betaflight (2022).
46. Bauersfeld, L. & Scaramuzza, D. Range, endurance, and optimal speed estimates for multicopters. *IEEE Robot. Autom. Lett.* **7**, 2953–2960 (2022).
47. Zhou, Y., Barnes, C., Lu, J., Yang, J. & Li, H. in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* 5745–5753 (IEEE, 2019).
48. Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W. & Abbeel, P. in *Proc. Robotics: Science and Systems XIV* (MIT Press Journals, 2018).
49. Molchanov, A. et al. in *Proc. 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* 59–66 (IEEE, 2019).
50. Andrychowicz, O. M. et al. Learning dexterous in-hand manipulation. *Int. J. Robot. Res.* **39**, 3–20 (2020).
51. Guadarrama, S. et al. TF-Agents: a library for reinforcement learning in TensorFlow. https://github.com/tensorflow/agents (2018).
52. Torrente, G., Kaufmann, E., Foehn, P. & Scaramuzza, D. Data-driven MPC for quadrotors. *IEEE Robot. Autom. Lett.* **6**, 3769–3776 (2021).
53. Ronneberger, O., Fischer, P. & Brox, T. in *Proc. International Conference on Medical Image Computing and Computer-assisted Intervention* 234–241 (Springer, 2015).
54. Intel RealSense T265 series product family. https://www.intelrealsense.com/wp-content/uploads/2019/09/Intel_RealSense_Tracking_Camera_Datasheet_Rev004_release.pdf (2019).
55. Ryou, G., Tal, E. & Karaman, S. Multi-fidelity black-box optimization for time-optimal quadrotor maneuvers. *Int. J. Robot. Res.* **40**, 1352–1369 (2021).
56. Pham, H. & Pham, Q.-C. A new approach to time-optimal path parameterization based on reachability analysis. *IEEE Trans. Robot.* **34**, 645–659 (2018).
57. Song, Y., Romero, A., Müller, M., Koltun, V. & Scaramuzza, D. Reaching the limit in autonomous racing: optimal control versus reinforcement learning. *Sci. Robot.* (in the press).
58. Foehn, P. et al. Agilicious: open-source and open-hardware agile quadrotor for vision-based flight. *Sci. Robot.* **7**, eabl6259 (2022).
59. Jones, E. S. & Soatto, S. Visual-inertial navigation, mapping and localization: a scalable real-time causal approach. *Int. J. Robot. Res.* **30**, 407–430 (2011).
