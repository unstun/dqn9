# 深度视觉运动策略的端到端训练

**原文**: End-to-End Training of Deep Visuomotor Policies

**作者**: Sergey Levine†, Chelsea Finn†, Trevor Darrell, Pieter Abbeel

**机构**: 加州大学伯克利分校计算机科学系

†共同第一作者

**期刊**: Journal of Machine Learning Research 17 (2016) 1-40

**编辑**: Jan Peters

**关键词**: 强化学习（Reinforcement Learning）、最优控制（Optimal Control）、视觉（Vision）、神经网络（Neural Networks）

---

## 摘要

策略搜索（policy search）方法能够让机器人学习各种任务的控制策略，但策略搜索的实际应用通常需要手工设计的感知、状态估计和低层控制组件。本文旨在回答以下问题：将感知系统和控制系统联合端到端训练，是否比分别训练每个组件能获得更好的性能？为此，我们开发了一种方法，可以学习将原始图像观测直接映射到机器人电机力矩的策略。该策略由具有 92,000 个参数的深度卷积神经网络（CNN）表示，并使用引导策略搜索（guided policy search）方法进行训练。引导策略搜索将策略搜索转化为监督学习问题，其监督信号由一种简单的轨迹中心强化学习方法提供。我们在一系列需要视觉与控制紧密协调的真实操作任务上评估了该方法，例如将瓶盖拧到瓶子上，并展示了与多种先前策略搜索方法的仿真对比结果。

---

## 1. 引言

在人类控制下，机器人可以完成令人印象深刻的任务，包括手术 [Lanfranco et al., 2004] 和家务劳动 [Wyrobek et al., 2008]。然而，即使对于基本任务，设计自主运行的感知和控制软件仍然是一个重大挑战。策略搜索方法有望让机器人通过经验自动学习新行为 [Kober et al., 2010b; Deisenroth et al., 2011; Kalakrishnan et al., 2011; Deisenroth et al., 2013]。但是，使用这些方法学到的策略通常依赖大量手工设计的感知和控制组件，以便为策略提供更易管理的低维观测和动作表示。尤其是视觉系统可能很复杂且容易出错，而且在策略训练期间通常不会被改进，也不会针对任务目标进行调整。

本文旨在回答以下问题：如果将感知系统与控制策略联合训练，而非分别训练，我们能否获得更有效的感觉运动控制（sensorimotor control）策略？为了表示同时执行感知和控制的策略，我们使用深度神经网络。深度神经网络表示近年来在计算机视觉、语音识别甚至电子游戏等多个领域取得了广泛成功。然而，将深度神经网络用于真实世界的感觉运动策略，即将图像像素和关节角度映射到电机力矩的机器人控制器，面临许多独特的挑战。深度神经网络的成功应用通常依赖于大量数据和输出的直接监督，而这两者在机器人控制中都不可用。真实世界的机器人交互数据很稀缺，且任务完成由高层代价函数定义，这意味着学习算法必须自行确定在每个时间点应采取的动作。从控制的角度来看，另一个复杂之处在于机器人传感器的观测并不能提供系统的完整状态。相反，任务相关对象的位置等重要状态信息必须从相机图像等输入中推断。

我们通过开发一种用于感觉运动深度学习（sensorimotor deep learning）的引导策略搜索算法以及一种专为机器人控制设计的新型 CNN 架构来应对这些挑战。引导策略搜索通过迭代地使用高效的无模型轨迹优化过程构建训练数据，将策略搜索转化为监督学习。我们证明这可以形式化为 Bregman ADMM（BADMM）[Wang and Banerjee, 2014] 的一个实例，并可用于证明算法收敛到局部最优解。在我们的方法中，系统的完整状态在训练时可观测，但在测试时不可观测。对于大多数任务，提供完整状态只需在训练期间的每次试验中将物体放置在若干已知位置之一。在测试时，学到的 CNN 策略可以处理新的未知配置，不再需要完整状态信息。由于策略通过监督学习优化，我们可以使用随机梯度下降等标准方法进行训练。我们的 CNN 有 92,000 个参数和 7 层，包括一种新型空间特征点变换（spatial feature point transformation），提供精确的空间推理并减少过拟合。这使我们能够用相对少量的数据和仅数十分钟的真实世界交互时间来训练策略。

我们通过在 PR2 机器人上学习以下策略来评估我们的方法：将积木插入形状分类盒、将瓶盖拧到瓶子上、用不同握法将玩具锤的爪部放到钉子下方、以及将衣架挂到架子上（见图 1）。这些任务需要定位、视觉跟踪以及处理复杂的接触动力学。我们的结果表明，与分别训练视觉和控制组件相比，端到端训练视觉运动策略（visuomotor policy）在一致性和泛化能力方面有所改善。我们还展示了仿真对比，表明在训练高维神经网络策略时，引导策略搜索优于多种先前方法。

> **图 1**: 本方法学到的视觉运动策略直接使用相机图像观测（左）来设置 PR2 机器人的电机力矩（右）。展示了衣架、积木、锤子和瓶子四个任务。

---

## 2. 相关工作

强化学习和策略搜索方法 [Gullapalli, 1990; Williams, 1992] 已被应用于机器人领域的多种任务，包括乒乓球 [Kober et al., 2010b]、物体操作 [Gullapalli, 1995; Peters and Schaal, 2008; Kober et al., 2010a; Deisenroth et al., 2011; Kalakrishnan et al., 2011]、行走运动 [Benbrahim and Franklin, 1997; Kohl and Stone, 2004; Tedrake et al., 2004; Geng et al., 2006; Endo et al., 2008] 和飞行 [Ng et al., 2004]。近年来有多篇综述介绍了机器人领域的策略搜索 [Deisenroth et al., 2013; Kober et al., 2013]。这些方法通常应用于机器人控制流水线的某个组件，该组件通常建立在手工设计的控制器（如 PD 控制器）之上，并接受处理后的输入，例如来自现有视觉流水线的输入 [Kalakrishnan et al., 2011]。我们的方法学习的策略将视觉输入和关节编码器读数直接映射到机器人关节的力矩。通过学习从感知到控制的整个映射，感知层可以针对任务性能进行调整，控制层也可以适应不完美的感知。

我们用卷积神经网络（CNN）表示策略。CNN 在计算机视觉和深度学习领域有着悠久的历史 [Fukushima, 1980; LeCun et al., 1989; Schmidhuber, 2015]，近年来因在多项视觉基准测试上的出色结果而备受瞩目 [Ciresan et al., 2011; Krizhevsky et al., 2012; Ciresan et al., 2012; Girshick et al., 2014a; Tompson et al., 2014; LeCun et al., 2015; He et al., 2015]。大多数 CNN 应用关注分类任务，通过连续的池化层丢弃位置信息以提供不变性 [Lee et al., 2009]。定位应用通常使用滑动窗口 [Girshick et al., 2014a] 或目标提案 [Endres and Hoiem, 2010; Uijlings et al., 2013; Girshick et al., 2014b] 来定位对象，从而将任务简化为分类。许多之前的机器人 CNN 应用并不直接考虑控制，而是将 CNN 用于更大机器人系统的感知组件 [Hadsell et al., 2009; Sung et al., 2015; Lenz et al., 2015b; Pinto and Gupta, 2015]。我们使用一种新型的 CNN 架构，能够自动学习捕捉场景空间信息的特征点，除了机器人编码器和相机的信息外，不需要任何额外监督。

深度学习在机器人控制中的应用近年来不如在视觉识别中那么普遍。通过动力学和图像形成过程进行反向传播通常不可行，因为它们通常不可微分，而且这种长距离反向传播可能导致极端的数值不稳定性。这个问题在循环神经网络的相关背景下也有观察到 [Hochreiter et al., 2001; Pascanu and Bengio, 2012]。网络的高维性也使得强化学习变得困难 [Deisenroth et al., 2013]。早期关于神经网络控制的开创性工作使用小型简单网络 [Pomerleau, 1989; Hunt et al., 1992; Bekey and Goldberg, 1992; Lewis et al., 1998; Bakker et al., 2003; Mayer et al., 2006]，并已在很大程度上被精心设计的、可通过强化学习高效学习的策略所取代 [Kober et al., 2013]。近年来关于感觉运动深度学习的工作处理了简单的任务空间运动 [Lenz et al., 2015a; Lampe and Riedmiller, 2013]，并使用无监督学习从图像获取低维状态空间 [Lange et al., 2012]。CNN 也被训练用于通过 Q-learning、蒙特卡洛树搜索和随机搜索来玩电子游戏 [Mnih et al., 2013; Koutník et al., 2013; Guo et al., 2014]，并应用于简单的仿真控制任务 [Watter et al., 2015; Lillicrap et al., 2015]。然而，这些方法只在缺乏真实世界视觉复杂性的合成域上得到验证，并且需要对于真实机器人学习来说不切实际的大量样本。我们的方法样本高效，只需几分钟的交互时间。据我们所知，这是第一种能够用直接力矩控制训练深度视觉运动策略来完成复杂高维操作技能的方法。

在真实机器人上学习视觉运动策略需要处理复杂的观测和高维策略表示。我们使用引导策略搜索来应对这些挑战。在引导策略搜索中，策略通过监督学习进行优化，该方法随策略维度优雅地扩展。监督学习的训练集可以使用已知动力学下的轨迹优化 [Levine and Koltun, 2013a,b, 2014; Mordatch and Todorov, 2014] 和在未知动力学下运行的轨迹中心强化学习方法 [Levine and Abbeel, 2014; Levine et al., 2015] 来构建，本文采用后者。在两种情况下，监督信号都会适应策略，以确保最终策略能够复现训练数据。

我们方法的目标也类似于视觉伺服（visual servoing），即在相机图像中的特征点上执行反馈控制 [Espiau et al., 1992; Mohta et al., 2014; Wilson et al., 1996]。然而，我们的视觉运动策略完全从真实世界数据中学习，不需要手工指定特征点或反馈控制器。这使得我们的方法在如何使用视觉信号方面具有更大的灵活性。我们的方法也不需要任何形式的相机标定。

---

## 3. 背景与概述

本节定义视觉运动策略学习问题，并概述我们方法的整体框架。核心组件是引导策略搜索算法，它将学习视觉运动策略的问题分解为独立的监督学习和轨迹学习阶段，每个阶段都比直接优化策略更容易。我们还讨论了适合端到端视觉和控制学习的策略架构，以及允许我们的方法应用于真实机器人平台的训练设置。

### 3.1 定义与问题表述

在策略搜索中，目标是学习策略 $\pi_\theta(u_t|o_t)$，使智能体能够根据观测 $o_t$ 选择动作 $u_t$ 来控制动力系统（如机器人）。策略来自某个由 $\theta$ 参数化的类别，例如 $\theta$ 可以是神经网络的权重。

系统由状态 $x_t$、动作 $u_t$ 和观测 $o_t$ 定义。例如，$x_t$ 可能包括机器人的关节角度、世界中物体的位置及其时间导数，$u_t$ 可能由电机力矩命令组成，$o_t$ 可能包括来自机器人机载相机的图像。本文处理有限时间幅度的情节式任务（episodic task），$t \in [1, \ldots, T]$。状态根据系统动力学 $p(x_{t+1}|x_t, u_t)$ 随时间演化，观测通常是状态的随机结果，服从 $p(o_t|x_t)$。动力学和观测分布都不假定为已知。

为了记号方便，我们用 $\pi_\theta(u_t|x_t)$ 表示策略条件于状态的动作分布。由于策略以观测 $o_t$ 为条件，该分布实际上由 $\pi_\theta(u_t|x_t) = \int \pi_\theta(u_t|o_t)p(o_t|x_t)do_t$ 给出。动力学和 $\pi_\theta(u_t|x_t)$ 共同诱导轨迹 $\tau = \{x_1, u_1, x_2, u_2, \ldots, x_T, u_T\}$ 上的分布：

$$\pi_\theta(\tau) = p(x_1) \prod_{t=1}^{T} \pi_\theta(u_t|x_t) p(x_{t+1}|x_t, u_t)$$

任务目标由代价函数 $\ell(x_t, u_t)$ 给出，策略搜索的目标是最小化期望 $E_{\pi_\theta(\tau)}[\sum_{t=1}^{T} \ell(x_t, u_t)]$，简记为 $E_{\pi_\theta(\tau)}[\ell(\tau)]$。

> **表 1**: 论文中常用符号汇总，包括状态 $x_t$、动作 $u_t$、观测 $o_t$、轨迹 $\tau$、代价函数 $\ell(x_t, u_t)$、未知系统动力学 $p(x_{t+1}|x_t, u_t)$、未知观测分布 $p(o_t|x_t)$、全局策略 $\pi_\theta(u_t|o_t)$、局部线性高斯控制器 $p_i(u_t|x_t)$ 等。

### 3.2 方法概述

我们的方法包含两个主要组件（如图 3 所示）。第一个是监督学习算法，训练形如 $\pi_\theta(u_t|o_t) = \mathcal{N}(\mu_\pi(o_t), \Sigma_\pi(o_t))$ 的策略，其中 $\mu_\pi(o_t)$ 和 $\Sigma_\pi(o_t)$ 都是一般的非线性函数。在我们的实现中，$\mu_\pi(o_t)$ 是一个深度卷积神经网络，而 $\Sigma_\pi(o_t)$ 是与观测无关的学习协方差。第二个组件是轨迹中心强化学习（trajectory-centric RL）算法，生成引导分布 $p_i(u_t|x_t)$，为训练策略提供监督。这两个组件构成了一个策略搜索算法，可以仅使用高层代价函数 $\ell(x_t, u_t)$ 来学习复杂的机器人任务。在训练期间，只有引导分布 $p_i(u_t|x_t)$ 的样本通过在物理系统上运行 rollout 来生成，从而避免了在物理硬件上执行部分训练的神经网络策略的需要。

一般来说，监督学习不会产生具有良好长期性能的策略，因为策略的微小错误会将系统带入训练数据分布之外的状态，导致误差累积。为避免这个问题，训练数据必须来自策略自身的状态分布 [Ross et al., 2011]。我们通过交替进行轨迹中心强化学习和监督学习来实现这一点。强化学习阶段适应当前策略 $\pi_\theta(u_t|o_t)$，在迭代过程中将监督提供的状态逐步靠近策略访问的状态。这被形式化为 BADMM 算法 [Wang and Banerjee, 2014] 的一个变体。

> **图 2**: 视觉运动策略架构。网络包含三个卷积层，之后是空间 softmax（spatial softmax）和期望位置层，将逐像素特征转换为特征点，更适合空间计算。特征点与机器人配置拼接后，通过三个全连接层产生力矩输出。网络包含约 92,000 个参数。

> **图 3**: 方法流程图，包括引导策略搜索的主要阶段和初始化阶段。初始化包括：收集视觉位姿数据、训练位姿 CNN、学习初始局部控制器，最后进行引导策略搜索的端到端训练。

为了减少训练视觉运动策略所需的经验量，我们引入了一种预训练方案。预训练的直觉是，虽然我们最终寻求的是同时结合视觉和控制的感觉运动策略，但视觉的低层方面可以独立初始化。为此，我们通过预测观测 $o_t$ 中未提供的 $x_t$ 元素（如场景中物体的位置）来预训练网络的卷积层。我们还会在卷积网络之外独立训练引导轨迹分布 $p_i(u_t|x_t)$，直到轨迹达到基本的任务完成能力，然后切换到带端到端 $\pi_\theta(u_t|o_t)$ 训练的完整引导策略搜索。在实现中，我们还用 Szegedy et al. [2014] 在 ImageNet [Deng et al., 2009] 分类上训练的模型初始化第一层滤波器。

---

## 4. 基于 BADMM 的引导策略搜索

引导策略搜索将策略搜索转化为监督学习问题，训练集由简单的轨迹中心强化学习算法生成。该算法优化线性高斯控制器（linear-Gaussian controller）$p_i(u_t|x_t)$，我们将 $p_i(u_t|x_t)$ 诱导的轨迹分布记为 $p_i(\tau)$。每个 $p_i(u_t|x_t)$ 从不同的初始状态出发成功执行。

最终策略 $\pi_\theta(u_t|o_t)$ 仅提供完整状态 $x_t$ 的观测 $o_t$，且动力学假定为未知。在外层循环中，通过在物理系统上运行相应控制器 $p_i(u_t|x_t)$ 来绘制样本轨迹 $\{\tau_i^j\}$。这些样本用于拟合动力学 $p_i(x_{t+1}|x_t, u_t)$ 来改进 $p_i(u_t|x_t)$，并作为策略的训练数据。内层循环交替优化每个 $p_i(\tau)$ 和优化策略以匹配这些轨迹分布。

### 4.1 算法推导

策略搜索方法最小化期望代价 $E_{\pi_\theta}[\ell(\tau)]$。我们首先将期望代价最小化改写为约束问题：

$$\min_{p, \pi_\theta} E_p[\ell(\tau)] \quad \text{s.t.} \quad p(u_t|x_t) = \pi_\theta(u_t|x_t) \quad \forall x_t, u_t, t \tag{1}$$

其中 $p(\tau)$ 称为引导分布（guiding distribution）。该表述与原问题等价，因为约束迫使两个分布相同。如果我们用样本 $x_1^i$ 近似初始状态分布 $p(x_1)$，则可以选择 $p(\tau)$ 为比 $\pi_\theta$ 更容易优化的分布类。

约束问题可通过对偶下降法（dual descent）求解，交替最小化拉格朗日函数和递增拉格朗日乘子。我们使用基于 BADMM [Wang and Banerjee, 2014] 的对偶下降方法，BADMM 是 ADMM [Boyd et al., 2011] 的变体，用 Bregman 散度增广拉格朗日函数。我们使用 KL 散度作为 Bregman 约束。BADMM 增广拉格朗日函数为：

$$L_\theta(\theta, p) = \sum_{t=1}^{T} E_{p(x_t, u_t)}[\ell(x_t, u_t)] + E_{p(x_t)\pi_\theta(u_t|x_t)}[\lambda_{x_t, u_t}] - E_{p(x_t, u_t)}[\lambda_{x_t, u_t}] + \nu_t \phi_t^\theta(\theta, p)$$

$$L_p(p, \theta) = \sum_{t=1}^{T} E_{p(x_t, u_t)}[\ell(x_t, u_t)] + E_{p(x_t)\pi_\theta(u_t|x_t)}[\lambda_{x_t, u_t}] - E_{p(x_t, u_t)}[\lambda_{x_t, u_t}] + \nu_t \phi_t^p(\theta, p)$$

其中 KL 散度期望项为：

$$\phi_t^p(p, \theta) = E_{p(x_t)}[D_{KL}(p(u_t|x_t) \| \pi_\theta(u_t|x_t))]$$

$$\phi_t^\theta(\theta, p) = E_{p(x_t)}[D_{KL}(\pi_\theta(u_t|x_t) \| p(u_t|x_t))]$$

使用一阶矩约束简化后，交替优化变为：

$$\theta \leftarrow \arg\min_\theta \sum_{t=1}^{T} E_{p(x_t)\pi_\theta(u_t|x_t)}[u_t^T \lambda_{\mu t}] + \nu_t \phi_t^\theta(\theta, p) \tag{2}$$

$$p \leftarrow \arg\min_p \sum_{t=1}^{T} E_{p(x_t, u_t)}[\ell(x_t, u_t) - u_t^T \lambda_{\mu t}] + \nu_t \phi_t^p(p, \theta) \tag{3}$$

$$\lambda_{\mu t} \leftarrow \lambda_{\mu t} + \alpha \nu_t (E_{\pi_\theta(u_t|x_t)p(x_t)}[u_t] - E_{p(u_t|x_t)p(x_t)}[u_t])$$

### 4.2 未知动力学下的轨迹优化

由于 $p(\tau)$ 为高斯分布，条件分布 $p(x_{t+1}|x_t, u_t)$ 和 $p(u_t|x_t)$ 是时变线性高斯的：

$$p(u_t|x_t) = \mathcal{N}(K_t x_t + k_t, C_t)$$

$$p(x_{t+1}|x_t, u_t) = \mathcal{N}(f_{x_t} x_t + f_{u_t} u_t + f_{c_t}, F_t)$$

这类控制器可以用少量真实世界样本高效学习。在未知动力学情况下，我们将 $p(x_{t+1}|x_t, u_t)$ 拟合到前一次迭代轨迹分布的样本。为避免估计的动力学偏差过大导致优化发散，我们通过 KL 散度约束限制分布变化：

$$\min_{p(\tau) \in \mathcal{N}(\tau)} L_p(p, \theta) \quad \text{s.t.} \quad D_{KL}(p(\tau) \| \hat{p}(\tau)) \leq \epsilon$$

该问题可使用对偶梯度下降高效求解，动力学通过在机器人上运行前一个控制器 $\hat{p}(u_t|x_t)$ 收集的样本拟合。拟合全局高斯混合模型（GMM）到元组 $(x_t, u_t, x_{t+1})$ 并用作先验，可大大减少样本复杂度。

### 4.3 监督策略优化

对于条件高斯策略 $\pi_\theta(u_t|o_t) = \mathcal{N}(\mu_\pi(o_t), \Sigma_\pi(o_t))$，目标函数为：

$$L_\theta(\theta, p) = \frac{1}{2N} \sum_{i=1}^{N} \sum_{t=1}^{T} E_{p_i(x_t, o_t)} \left[ \text{tr}[C_{ti}^{-1} \Sigma_\pi(o_t)] - \log|\Sigma_\pi(o_t)| + (\mu_\pi(o_t) - \mu_{ti}^p(x_t))^T C_{ti}^{-1} (\mu_\pi(o_t) - \mu_{ti}^p(x_t)) + 2\lambda_{\mu t}^T \mu_\pi(o_t) \right]$$

其中 $\mu_{ti}^p(x_t)$ 是 $p_i(u_t|x_t)$ 的均值，$C_{ti}$ 是协方差，期望使用每个 $p_i(\tau)$ 的样本及其对应的观测 $o_t$ 来评估。$L_\theta(\theta, p)$ 本质上是策略均值与轨迹分布均值动作之间的加权二次损失，偏移量为拉格朗日乘子，权重是轨迹分布条件分布的精度矩阵。

我们使用随机梯度下降（SGD）优化 $L_\theta(\theta, p)$。为了让复杂的神经网络有足够多的样本，我们发现将前几次迭代的采样观测也纳入策略优化是有益的，并使用重要性采样进行加权。

### 4.4 与先前引导策略搜索方法的比较

BADMM 公式是本文的新贡献。相比 Levine and Koltun [2014] 提出的类似公式，BADMM 在轨迹优化阶段求解凸问题，因此速度更快且更易实现，特别是在轨迹数量 $p_i(\tau)$ 较多时。

---

## 5. 端到端视觉运动策略

### 5.1 视觉运动策略架构

我们的视觉运动策略在机器人上以 20 Hz 运行，将单目 RGB 图像和机器人配置映射到 7 自由度手臂的关节力矩。配置包括关节角度和末端执行器的位姿（由末端执行器空间中的 3 个点定义）及其速度，但不包括目标物体或目标的位置，这些必须从图像中确定。

我们的策略不使用池化（pooling），因为位置信息对控制很重要。此外，我们提出了一种新型 CNN 架构，能够在无需图像空间直接监督的情况下从图像中估计空间信息。

网络架构如图 2 所示。视觉处理层由三个卷积层组成，每层学习应用于以每个像素为中心的图像块的滤波器组。每个卷积层后跟一个整流非线性。第三个卷积层包含 32 个分辨率为 $109 \times 109$ 的响应图。这些响应图通过空间 softmax 函数处理：

$$s_{cij} = \frac{e^{a_{cij}}}{\sum_{i'j'} e^{a_{ci'j'}}}$$

每个 softmax 输出通道是图像中某个特征位置的概率分布。为了将该分布转换为坐标表示 $(f_{cx}, f_{cy})$，网络计算每个特征的期望图像位置：

$$f_{cx} = \sum_{ij} s_{cij} x_{ij}, \quad f_{cy} = \sum_{ij} s_{cij} y_{ij}$$

其中 $(x_{ij}, y_{ij})$ 是响应图中点 $(i, j)$ 的图像空间位置。空间 softmax 和期望位置计算的组合实现了一种软 argmax（soft-argmax）。空间特征点 $(f_{cx}, f_{cy})$ 与机器人的配置拼接后，送入两个各含 40 个整流单元的全连接层，最后通过线性连接输出力矩。完整网络包含约 92,000 个参数，其中约 86,000 个在卷积层中。

空间 softmax 还提供侧抑制（lateral inhibition），抑制低的错误激活，只保留强激活。这使我们的策略对干扰物更加鲁棒，提供了对新视觉变化的泛化能力。

### 5.2 视觉运动策略训练

引导策略搜索的轨迹优化阶段使用系统的完整状态，但最终策略只使用观测。这种工具化训练（instrumented training）是许多机器人任务的自然选择。在我们的任务中，未观测变量是目标物体的位姿。训练期间，目标物体通常由机器人的左手夹持器固定，右臂执行任务，这允许机器人将目标移动到一系列已知位置。

为加速学习，我们通过以下方式预训练：
1. **视觉层预训练**：机器人将目标物体移动到一系列随机位置，记录相机图像和物体位姿。该数据集用于训练位姿回归 CNN。我们使用 1000 张从随机手臂运动中收集的图像，并用 Szegedy et al. [2014] 的 ImageNet 预训练模型初始化第一层滤波器。
2. **轨迹预训练**：在不优化视觉运动策略的情况下进行 15 次引导策略搜索迭代，使用接收完整状态的小型网络来约束轨迹。

初始化后，使用引导策略搜索训练完整的视觉运动策略。在监督策略优化阶段，先单独优化全连接运动控制层，然后进行整个网络的端到端优化。

---

## 6. 实验评估

本节通过一系列实验回答以下问题：
1. 引导策略搜索与其他策略搜索方法在训练复杂高维策略时如何比较？
2. 轨迹优化算法能否在未知动力学的真实机器人平台上用于多种任务？
3. 空间 softmax 架构与更标准的 CNN 架构相比表现如何？
4. 在视觉运动策略中联合端到端训练感知和控制系统是否比分别训练每个组件提供更好的性能？

### 6.1 与先前策略搜索方法的仿真对比

在仿真机器人控制任务上将本方法与先前策略搜索技术进行比较，包括 2D 和 3D 钉插入、章鱼臂控制、平面游泳和行走。

**对比方法**: REPS [Peters et al., 2010]、奖励加权回归（RWR）[Peters and Schaal, 2007; Kober and Peters, 2009]、交叉熵方法（CEM）[Rubinstein and Kroese, 2004]、PILCO [Deisenroth and Rasmussen, 2011]，以及使用已知模型的 iLQG [Li and Todorov, 2004] 作为基线。

> **图 4**: 2D 和 3D 钉插入、章鱼臂和游泳任务的线性高斯控制器学习结果。本方法使用更少的样本找到了比先前方法更好的解，GMM 进一步减少了所需样本量。

> **图 5**: 神经网络策略对比。在插入任务中，策略在四个槽位置上训练，用虚线显示对新位置的泛化。本方法能够获得在训练和测试槽位置都成功的策略。

**高斯轨迹分布结果**: 本方法用更少的样本学到了更有效的控制器。在 3D 插入任务上，甚至优于使用已知模型的 iLQG 基线。

**神经网络策略结果**: 只有本方法能够获得成功的策略来定位训练和测试槽位置。这些对比表明，即使对于中等大小的神经网络策略，使用有限样本进行连续控制的训练对许多先前策略搜索算法来说也非常困难。

### 6.2 在 PR2 机器人上学习线性高斯控制器

在真实 PR2 机器人上展示了多种操作任务的学习能力。任务包括堆叠乐高积木、将木环穿到钉子上、为玩具飞机装轮子、将鞋撑插入鞋中、拧药瓶盖和水瓶盖。

> **图 6**: 线性高斯控制器评估的任务：(a) 在固定基座上堆叠乐高积木，(b) 堆在独立积木上，(c) 双手夹持积木；(d) 将木环穿到钉子上；(e) 为玩具飞机装轮子；(f) 将鞋撑插入鞋中；(g,h) 拧药瓶盖，(i) 拧水瓶盖。

> **图 7**: 线性高斯控制器训练期间到目标点的距离学习曲线。

每个控制器学习所需的样本数约为 20-25 个，远低于文献中许多先前策略搜索方法。总学习时间约为每个任务十分钟。

> **表 2**: 目标物体扰动下线性高斯控制器的成功率。所有控制器对 1 cm 扰动都具有鲁棒性，在 2 cm 扰动下也经常成功。

### 6.3 空间 Softmax CNN 架构评估

与更标准的卷积网络在位姿估计预训练任务上进行比较。

> **表 3**: 不同架构的平均位姿估计精度（厘米）：
> - softmax + 特征点（本文）: $1.30 \pm 0.73$
> - softmax + 全连接层: $2.59 \pm 1.19$
> - 全连接层: $4.75 \pm 2.29$
> - 最大池化 + 全连接层: $3.71 \pm 1.73$

使用 softmax 和期望算子显著提高了位姿估计精度，因为它迫使网络学习特征点，提供了适合空间推理的简洁表示。

### 6.4 深度视觉运动策略评估

在 PR2 机器人上对完整视觉运动策略训练算法进行评估。

**实验任务**: 将衣架挂到架子上、将积木插入形状分类盒、用不同握法将锤子爪部放到钉子下方、拧瓶盖。

> **图 8**: 视觉运动策略实验中的任务示意，展示了衣架、形状盒、锤子任务中目标位置的变化，以及锤子任务中的不同握法。

**实验条件**: (1) 训练目标位置和握法，(2) 训练中未见过的新目标位置（空间测试），(3) 训练位置加视觉干扰物（视觉测试）。

**对比方法**:
- **端到端训练**（本文方法）
- **位姿特征基线**: 丢弃位姿预测器的最后一层，使用特征点
- **位姿预测基线**: 将预测的位姿输入控制层，类似于标准模块化方法

> **图 9**: 实验结果。成功率如下：
>
> | 任务 | 条件 | 端到端 | 位姿特征 | 位姿预测 |
> |------|------|--------|----------|----------|
> | 衣架 | 训练(18) / 空间(24) / 视觉(18) | 100% / 100% / 100% | 88.9% / 87.5% / 83.3% | 55.6% / 58.3% / 66.7% |
> | 形状盒 | 训练(27) / 空间(36) / 视觉(40) | 96.3% / 91.7% / 87.5% | 70.4% / 83.3% / 40% | 0% / 0% / n/a |
> | 锤子 | 训练(45) / 空间(60) / 视觉(60) | 91.1% / 86.7% / 78.3% | 62.2% / 75.0% / 53.3% | 8.9% / 18.3% / n/a |
> | 瓶盖 | 训练(27) / 空间(12) / 视觉(40) | 88.9% / 83.3% / 62.5% | 55.6% / 58.3% / 27.5% | - / - / - |

结果表明，完整的端到端训练显著优于分别训练视觉和控制组件。位姿预测基线性能较差，因为许多任务的容差仅为几毫米，而位姿估计精度约为 1 厘米。端到端训练使策略能够学习改善机器人精度的视觉特征和控制策略。

**视觉干扰物**: 策略对与目标物体视觉分离的干扰物表现出一定的容忍度。这部分得益于空间 softmax 的侧抑制效应，能够抑制非最大激活。

### 6.5 端到端训练学到的特征

> **图 10**: 四个任务中策略在执行期间跟踪的特征点。每个特征点显示为不同的随机颜色。策略在目标物体和机器人手爪及手臂上找到特征。在瓶盖任务中，策略正确忽略了背景中的干扰瓶子。

> **图 11**: 每个任务学到的特征点比较。蓝色为策略产生的特征点，红色为位姿预测网络的特征点。端到端训练的策略倾向于在目标物体和机器人手臂上发现更多特征点。

端到端训练后，策略获得了与位姿预测 CNN 初始化时截然不同的特征点集合。端到端训练的模型在任务相关物体上找到更多特征点，在背景物体上找到更少的点，表明策略通过获取目标驱动的视觉特征来提升性能。

### 6.6 计算性能与样本效率

我们使用 Caffe 深度学习库进行 CNN 训练。每个视觉运动策略总共需要 3-4 小时的训练时间：20-30 分钟用于机器人上的位姿预测数据收集，40-60 分钟用于完全观测的轨迹预训练（可并行进行），1.5-2.5 小时用于引导策略搜索的端到端训练。

> **表 4**: 每个视觉运动策略所使用的试验次数：
>
> | 任务 | 轨迹预训练 | 端到端训练 | 总计 |
> |------|-----------|-----------|------|
> | 衣架 | 120 | 36 | 156 |
> | 形状盒 | 90 | 81 | 171 |
> | 锤子 | 150 | 90 | 240 |
> | 瓶盖 | 180 | 108 | 288 |

每次试验 5 秒，仅约 15 分钟的训练时间涉及在机器人上执行试验。

---

## 7. 讨论与未来工作

本文提出了一种学习使用单目相机原始输入的机器人控制策略的方法。这些策略由一种新型卷积神经网络架构表示，可以使用引导策略搜索算法端到端训练。该算法将策略搜索问题分解为使用完整状态信息的轨迹优化阶段和仅使用观测的监督学习阶段。实验结果表明，我们的方法可以执行复杂的操作技能，端到端训练比使用固定的位姿预测视觉层带来了显著的性能提升。

虽然我们展示了对场景变化的适度泛化，但当前方法不能泛化到显著不同的设置，特别是当视觉干扰物遮挡操作对象时。更实际的替代方案包括：同时在多个机器人上训练策略、开发更复杂的正则化和预训练技术、引入人工数据增强。

我们的方法利用了训练期间已知的完全可观状态空间。这既是弱点也是优势：它允许用很少的样本训练引导策略搜索的线性高斯控制器，但限制了方法的适用任务范围。解决这一限制的有前景方向是将我们的方法与无监督状态空间学习相结合 [Lange et al., 2012; Watter et al., 2015; Finn et al., 2015]。

未来工作中，我们希望探索更复杂的策略架构（如能通过保留过去观测的记忆来处理大量遮挡的循环策略），以及将方法扩展到更多可受益于视觉输入和其他丰富感觉模态（如触觉输入、听觉输入）的任务。

---

## 致谢

本研究部分由 DARPA 青年教师奖、陆军研究办公室 MAST 项目、NSF 奖项 IIS-1427425 和 IIS-1212798、伯克利视觉与学习中心以及伯克利 EECS 系奖学金资助。

---

## 附录 A. 引导策略搜索算法细节

### A.1 BADMM 对偶变量与权重调整

步长 $\alpha = 0.1$。权重 $\nu_t$ 初始化为 0.01，基于以下策略递增：在每次迭代中，计算所有时间步 $p(u_t|x_t)$ 和 $\pi_\theta(u_t|x_t)$ 之间的平均 KL 散度及其标准差。KL 散度高于平均值的时间步对应的 $\nu_t$ 增加 2 倍，低于平均值两个标准差的时间步对应的 $\nu_t$ 减少 2 倍。

### A.2 策略方差优化

高斯策略 $\pi_\theta(u_t|o_t)$ 的方差 $\Sigma_\pi$ 不依赖于观测。对 $L_\theta(\theta, p)$ 中仅依赖 $\Sigma_\pi$ 的项求导并令其为零，得到：

$$\Sigma_\pi = \left[ \frac{1}{NT} \sum_{i=1}^{N} \sum_{t=1}^{T} C_{ti}^{-1} \right]^{-1}$$

### A.3 动力学拟合

线性高斯动力学 $p(x_{t+1}|x_t, u_t) = \mathcal{N}(f_{x_t} x_t + f_{u_t} u_t + f_{c_t}, F_t)$ 通过对真实系统样本进行拟合获得。为减少样本复杂度，使用全局模型（高斯混合模型）作为先验。全局模型拟合到所有可用元组 $\{x_t^i, u_t^i, x_{t+1}^i\}$，然后用正态逆 Wishart（normal-inverse-Wishart）先验进行贝叶斯线性回归。

高斯混合模型特别适合关节刚体接触动力学系统，因为这类系统可以粗略建模为分段线性动力学，每个混合元素对应不同的线性模式。

### A.4 轨迹优化

约束轨迹优化问题：

$$\min_{p(\tau) \in \mathcal{N}(\tau)} L_p(p, \theta) \quad \text{s.t.} \quad D_{KL}(p(\tau) \| \hat{p}(\tau)) \leq \epsilon$$

可以用标准 LQR 后向传递求解对偶梯度下降中的原始优化。最优控制律为：

$$p(u_t|x_t) = \mathcal{N}(K_t x_t + k_t; Q_{u,u_t}^{-1})$$

其中 $K_t$、$k_t$ 和 $Q_{u,u_t}$ 可通过标准 LQR 后向传递获得，包括递归计算二次 Q 函数和值函数：

$$Q_{xu,xu_t} = \tilde{c}_{xu,xu_t} + f_{xu_t}^T V_{x,x_{t+1}} f_{xu_t}$$

$$V_{x,x_t} = Q_{x,x_t} - Q_{u,x_t}^T Q_{u,u_t}^{-1} Q_{u,x_t}$$

$$K_t = -Q_{u,u_t}^{-1} Q_{u,x_t}, \quad k_t = -Q_{u,u_t}^{-1} Q_{u_t}$$

---

## 附录 B. 实验设置细节

### B.1 仿真实验细节

所有仿真实验使用 MuJoCo 物理仿真 [Todorov et al., 2012]。

- **钉插入**: 2D 版本有 6 个状态维度和 2 个动作维度；3D 版本有 12 个状态维度。代价函数 $\ell(x_t, u_t) = \frac{1}{2} w_u \|u_t\|^2 + w_p \ell_{12}(p_{x_t} - p^\star)$。
- **章鱼臂**: 25 个自由度，50 维状态空间。
- **游泳**: 3 个连杆，5 个自由度，10 维状态空间。代价函数 $\ell(x_t, u_t) = \frac{1}{2} w_u \|u_t\|^2 + \frac{1}{2} w_v \|v_{x_{x_t}} - v_x^\star\|^2$。
- **行走**: 9 个自由度，18 维状态空间，6 个动作维度。

### B.2 机器人实验细节

所有机器人实验在 PR2 机器人上进行，控制频率 20 Hz，使用 PrimeSense Carmine 传感器的 RGB 相机，图像降采样至 $240 \times 240 \times 3$。每个回合 5 秒。代价函数为：

$$\ell(x_t, u_t) = w_{\ell_2} d_t^2 + w_{\log} \log(d_t^2 + \alpha) + w_u \|u_t\|^2$$

其中 $d_t$ 是末端执行器空间中三个点与其目标位置之间的距离。

各任务具体设置：
- **衣架**: 两种抓握角度（相差约 35°），三个不同距离的架子位置
- **形状盒**: 九个位置，分布在 $16 \text{cm} \times 10 \text{cm}$ 的矩形区域
- **锤子**: 三种抓握角度（每个相差 22.5°，总变化 45°），五个钉子位置
- **瓶盖**: 九个瓶子位置，分布在 $16 \text{cm} \times 10 \text{cm}$ 的矩形区域

---

## 参考文献

J. A. Bagnell and J. Schneider. Covariant policy search. In International Joint Conference on Artificial Intelligence (IJCAI), 2003.

B. Bakker, V. Zhumatiy, G. Gruener, and J. Schmidhuber. A robot that reinforcement-learns to identify and memorize important previous observations. In International Conference on Intelligent Robots and Systems (IROS), 2003.

G. Bekey and K. Goldberg. Neural Networks in Robotics. Springer US, 1992.

H. Benbrahim and J. A. Franklin. Biped dynamic walking using reinforcement learning. Robotics and Autonomous Systems, 22:283–302, 1997.

W. Böhmer, S. Grünewälder, Y. Shen, M. Musial, and K. Obermayer. Construction of approximation spaces for reinforcement learning. Journal of Machine Learning Research, 14(1):2067–2118, January 2013.

S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1122, 2011.

D. Ciresan, U. Meier, J. Masci, L. Gambardella, and J. Schmidhuber. Flexible, high performance convolutional neural networks for image classification. In International Joint Conference on Artificial Intelligence (IJCAI), 2011.

D. Ciresan, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. In Computer Vision and Pattern Recognition (CVPR), 2012.

M. Deisenroth and C. Rasmussen. PILCO: a model-based and data-efficient approach to policy search. In International Conference on Machine Learning (ICML), 2011.

M. Deisenroth, C. Rasmussen, and D. Fox. Learning to control a low-cost manipulator using data-efficient reinforcement learning. In Robotics: Science and Systems (RSS), 2011.

M. Deisenroth, G. Neumann, and J. Peters. A survey on policy search for robotics. Foundations and Trends in Robotics, 2(1-2):1–142, 2013.

J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition (CVPR), 2009.

G. Endo, J. Morimoto, T. Matsubara, J. Nakanishi, and G. Cheng. Learning CPG-based biped locomotion with a policy gradient method: Application to a humanoid robot. International Journal of Robotic Research, 27(2):213–228, 2008.

I. Endres and D. Hoiem. Category independent object proposals. In European Conference on Computer Vision (ECCV). 2010.

Y. Engel, P. Szabó, and D. Volkinshtein. Learning to control an octopus arm with Gaussian process temporal difference methods. In Advances in Neural Information Processing Systems (NIPS), 2005.

B. Espiau, F. Chaumette, and P. Rives. A new approach to visual servoing in robotics. IEEE Transactions on Robotics and Automation, 8(3), 1992.

C. Finn, X. Tan, Y. Duan, T. Darrell, S. Levine, and P. Abbeel. Learning visual feature spaces for robotic manipulation with deep spatial autoencoders. arXiv preprint arXiv:1509.06113, 2015.

K. Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36:193–202, 1980.

T. Geng, B. Porr, and F. Wörgötter. Fast biped walking with a reflexive controller and realtime policy searching. In Advances in Neural Information Processing Systems (NIPS), 2006.

R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2014a.

R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014b.

V. Gullapalli. A stochastic reinforcement learning algorithm for learning real-valued functions. Neural Networks, 3(6):671–692, 1990.

V. Gullapalli. Skillful control under uncertainty via direct reinforcement learning. Reinforcement Learning and Robotics, 15(4):237–246, 1995.

X. Guo, S. Singh, H. Lee, R. L. Lewis, and X. Wang. Deep learning for real-time Atari game play using offline Monte-Carlo tree search planning. In Advances in Neural Information Processing Systems (NIPS), 2014.

R. Hadsell, P. Sermanet, J. B. A. Erkan, and M. Scoffier. Learning long-range vision for autonomous off-road driving. Journal of Field Robotics, pages 120–144, 2009.

K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.

S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. In A Field Guide to Dynamic Recurrent Neural Networks. IEEE Press, 2001.

K. J. Hunt, D. Sbarbaro, R. Żbikowski, and P. J. Gawthrop. Neural networks for control systems: A survey. Automatica, 28(6):1083–1112, November 1992.

D. Jacobson and D. Mayne. Differential Dynamic Programming. Elsevier, 1970.

M. Jägersand, O. Fuentes, and R. C. Nelson. Experimental evaluation of uncalibrated visual servoing for precision manipulation. In International Conference on Robotics and Automation (ICRA), 1997.

Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv preprint arXiv:1408.5093, 2014.

S. Jodogne and J. H. Piater. Closed-loop learning of visual control policies. Journal of Artificial Intelligence Research, 28:349–391, 2007.

R. Jonschkowski and O. Brock. State representation learning in robotics: Using prior knowledge about physical interaction. In Proceedings of Robotics: Science and Systems, 2014.

M. Kalakrishnan, L. Righetti, P. Pastor, and S. Schaal. Learning force control policies for compliant manipulation. In International Conference on Intelligent Robots and Systems (IROS), 2011.

S. M. Khansari-Zadeh and A. Billard. BM: An iterative algorithm to learn stable non-linear dynamical systems with Gaussian mixture models. In International Conference on Robotics and Automation (ICRA), 2010.

J. Kober and J. Peters. Learning motor primitives for robotics. In International Conference on Robotics and Automation (ICRA), 2009.

J. Kober, K. Muelling, O. Kroemer, C.H. Lampert, B. Schoelkopf, and J. Peters. Movement templates for learning of hitting and batting. In International Conference on Robotics and Automation (ICRA), 2010a.

J. Kober, E. Oztop, and J. Peters. Reinforcement learning to adjust robot movements to new situations. In Robotics: Science and Systems (RSS), 2010b.

J. Kober, J. A. Bagnell, and J. Peters. Reinforcement learning in robotics: A survey. International Journal of Robotic Research, 32(11):1238–1274, 2013.

N. Kohl and P. Stone. Policy gradient reinforcement learning for fast quadrupedal locomotion. In International Conference on Robotics and Automation (IROS), 2004.

J. Koutník, G. Cuccu, J. Schmidhuber, and F. Gomez. Evolving large-scale neural networks for vision-based reinforcement learning. In Conference on Genetic and Evolutionary Computation, GECCO '13, 2013.

A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NIPS). 2012.

T. Lampe and M. Riedmiller. Acquiring visual servoing reaching and grasping skills using neural reinforcement learning. In International Joint Conference on Neural Networks (IJCNN), 2013.

A. Lanfranco, A. Castellanos, J. Desai, and W. Meyers. Robotic surgery: a current perspective. Annals of surgery, 239(1):14, 2004.

S. Lange, M. Riedmiller, and A. Voigtlaender. Autonomous reinforcement learning on raw visual input data in a real world application. In International Joint Conference on Neural Networks, 2012.

Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Handwritten digit recognition with a back-propagation network. In Advances in Neural Information Processing Systems (NIPS), 1989.

Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. Nature, 521:436–444, May 2015.

H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In International Conference on Machine Learning (ICML), 2009.

Ian Lenz, Ross Knepper, and Ashutosh Saxena. DeepMPC: Learning deep latent features for model predictive control. In RSS, 2015a.

Ian Lenz, Honglak Lee, and Ashutosh Saxena. Deep learning for detecting robotic grasps. IJRR, 2015b.

S. Levine and P. Abbeel. Learning neural network policies with guided policy search under unknown dynamics. In Advances in Neural Information Processing Systems (NIPS), 2014.

S. Levine and V. Koltun. Guided policy search. In International Conference on Machine Learning (ICML), 2013a.

S. Levine and V. Koltun. Variational policy search via trajectory optimization. In Advances in Neural Information Processing Systems (NIPS), 2013b.

S. Levine and V. Koltun. Learning complex neural network policies with trajectory optimization. In International Conference on Machine Learning (ICML), 2014.

S. Levine, N. Wagener, and P. Abbeel. Learning contact-rich manipulation skills with guided policy search. In International Conference on Robotics and Automation (ICRA), 2015.

F. L. Lewis, A. Yesildirak, and S. Jagannathan. Neural Network Control of Robot Manipulators and Nonlinear Systems. Taylor & Francis, Inc., 1998.

W. Li and E. Todorov. Iterative linear quadratic regulator design for nonlinear biological movement systems. In ICINCO (1), pages 222–229, 2004.

T. Lillicrap, J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

R. Lioutikov, A. Paraschos, G. Neumann, and J. Peters. Sample-based information-theoretic stochastic optimal control. In International Conference on Robotics and Automation, 2014.

H. Mayer, F. Gomez, D. Wierstra, I. Nagy, A. Knoll, and J. Schmidhuber. A system for robotic heart surgery that learns to tie knots using recurrent neural networks. In International Conference on Intelligent Robots and Systems (IROS), 2006.

W. Meeussen, M. Wise, S. Glaser, S. Chitta, C. McGann, P. Mihelich, E. Marder-Eppstein, M. Muja, Victor Eruhimov, T. Foote, J. Hsu, R.B. Rusu, B. Marthi, G. Bradski, K. Konolige, B. Gerkey, and E. Berger. Autonomous door opening and plugging in with a personal robot. In International Conference on Robotics and Automation (ICRA), 2010.

V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller. Playing Atari with deep reinforcement learning. NIPS '13 Workshop on Deep Learning, 2013.

K. Mohta, V. Kumar, and K. Daniilidis. Vision based control of a quadrotor for perching on planes and lines. In International Conference on Robotics and Automation (ICRA), 2014.

I. Mordatch and E. Todorov. Combining the benefits of function approximation and trajectory optimization. In Robotics: Science and Systems (RSS), 2014.

A. Y. Ng, H. J. Kim, M. I. Jordan, and S. Sastry. Inverted autonomous helicopter flight via reinforcement learning. In International Symposium on Experimental Robotics, 2004.

R. Pascanu and Y. Bengio. On the difficulty of training recurrent neural networks. Technical Report arXiv:1211.5063, Universite de Montreal, 2012.

B. Pepik, M. Stark, P. Gehler, and B. Schiele. Teaching 3D geometry to deformable part models. In Computer Vision and Pattern Recognition (CVPR), 2012.

J. Peters and S. Schaal. Applying the episodic natural actor-critic architecture to motor primitive learning. In European Symposium on Artificial Neural Networks (ESANN), 2007.

J. Peters and S. Schaal. Reinforcement learning of motor skills with policy gradients. Neural Networks, 21(4):682–697, 2008.

J. Peters, K. Mülling, and Y. Altün. Relative entropy policy search. In AAAI Conference on Artificial Intelligence, 2010.

Lerrel Pinto and Abhinav Gupta. Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours. CoRR, abs/1509.06825, 2015.

D. Pomerleau. ALVINN: an autonomous land vehicle in a neural network. In Advances in Neural Information Processing Systems (NIPS), 1989.

S. Ross, G. Gordon, and A. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. Journal of Machine Learning Research, 15:627–635, 2011.

S. Ross, N. Melik-Barkhudarov, K. Shaurya Shankar, A. Wendel, D. Dey, J. A. Bagnell, and M. Hebert. Learning monocular reactive UAV control in cluttered natural environments. In International Conference on Robotics and Automation (ICRA), 2013.

R. Rubinstein and D. Kroese. The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning. Springer, 2004.

S. Savarese and L. Fei-Fei. 3D generic object categorization, localization and pose estimation. In International Conference on Computer Vision (ICCV), 2007.

J. Schmidhuber. Deep learning in neural networks: An overview. Neural Networks, 61:85–117, 2015.

P. Y. Simard, D. Steinkraus, and J. C. Platt. Best practices for convolutional neural networks applied to visual document analysis. In Seventh International Conference on Document Analysis and Recognition, 2003.

F. Stulp and O. Sigaud. Path integral policy improvement with covariance matrix adaptation. In International Conference on Machine Learning (ICML), 2012.

Jaeyong Sung, Seok Hyun Jin, and Ashutosh Saxena. Robobarista: Object part based transfer of manipulation trajectories from crowd-sourcing in 3d pointclouds. CoRR, abs/1504.03071, 2015.

C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. arXiv preprint arXiv:1409.4842, 2014.

R. Tedrake, T. Zhang, and H. Seung. Stochastic policy gradient reinforcement learning on a simple 3d biped. In International Conference on Intelligent Robots and Systems (IROS), 2004.

E. Theodorou, J. Buchli, and S. Schaal. Reinforcement learning of motor skills in high dimensions. In International Conference on Robotics and Automation (ICRA), 2010.

E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.

J. J. Tompson, A. Jain, Y. LeCun, and C. Bregler. Joint training of a convolutional network and a graphical model for human pose estimation. In Advances in Neural Information Processing Systems (NIPS), 2014.

J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. International Journal of Computer Vision, 2013.

H. van Hoof, J. Peters, and G. Neumann. Learning of non-parametric control policies with high-dimensional state features. In International Conference on Artificial Intelligence and Statistics, 2015.

H. Wang and A. Banerjee. Bregman alternating direction method of multipliers. In Advances in Neural Information Processing Systems (NIPS). 2014.

M. Watter, J. Springenberg, J. Boedecker, and M. Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. In Advanced in Neural Information Processing Systems (NIPS), 2015.

R. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4):229–256, May 1992.

W. J. Wilson, C. W. Williams Hulls, and G. S. Bell. Relative end-effector control using cartesian position based visual servoing. IEEE Transactions on Robotics and Automation, 12(5), 1996.

K.A. Wyrobek, E.H. Berger, HF M. Van der Loos, and K. Salisbury. Towards a personal robotics development platform: Rationale and design of an intrinsically safe personal robot. In International Conference on Robotics and Automation (ICRA), 2008.

B. H. Yoshimi and P. K. Allen. Active, uncalibrated visual servoing. In International Conference on Robotics and Automation (ICRA), 1994.
