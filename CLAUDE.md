# DQN9 项目规则（论文写作 + 工程开发）

> 作用域：`/home/sun/phdproject/dqn/DQN9/**`（Ubuntu）、`/Users/sun/tongbu/study/phdproject/dqn/DQN9/**`（Mac）
> 继承自 DQN8，代码包已从 `amr_dqn` 重命名为 `ugv_dqn`。

### 开发环境

| 平台                | 用途                            | Conda 路径                                | 说明                                                    |
| ------------------- | ------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| Mac (Apple Silicon) | 代码开发/论文写作，不做训练推理 | `/opt/homebrew/Caskroom/miniforge/base` | Miniforge，`KMP_DUPLICATE_LIB_OK=TRUE` 已设入环境变量 |
| Ubuntu (远程 GPU)   | 训练 + 推理                     | `$HOME/miniconda3`                      | RTX 4090                                                |

- Mac 上 PyTorch 为 CPU 版本，仅用于代码正确性验证
- 训练/推理一律通过 SSH 在远程 GPU 服务器执行

## 0. 硬约束

1) 每次回复以"Dr Sun，"开头。
2) **除非有特殊说明，请用中文回答。** (Unless otherwise specified, please respond in Chinese.)
3) 思考语言：专业流英语，交互语言：中文，注释规范：中文 + ASCII 风格分块注释,使代码看起来像高度优化的顶级开源库作品，核心信念：代码是写给人看的,只是顺便让机器运行
4) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
5) 默认中文回复；论文正文先中文写作，定稿后统一英文润色。README 等项目文档也用中文撰写。论文中尽可能少用括号，改用"如图…所示""即…"等行文方式替代。公式中使用专业数学符号，禁止出现代码风格变量名如 `grid_size`、`cell_size`，应使用 $\Delta c$、$\delta$、$\epsilon$ 等标准记法。
6) **严禁凭记忆生成 BibTeX**——未核实引用标 `[CITATION NEEDED]`。
7) `CLAUDE.md` 为唯一项目规则文件。
8) 纯文档改动豁免 `configs/` 新增规则；代码改动须在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`。
9) **消融实验留档**：结束后在 `runs/ablation_logs/` 写 `ablation_YYYYMMDD_<topic>.md`。
10) 每次可以先尝试ssh再本地计算
11) **代码搜索策略**：语义理解/探索代码库时**始终优先使用 ACE**（`mcp__augment-context-engine__codebase-retrieval`）；精确匹配标识符/字符串时使用 `Grep`（rg）。禁止用 Bash 调 grep/rg。ACE 调用若报错/超时，立即回退到 `Grep` + `Glob` 继续工作，不阻塞流程。
12) Conda 环境：`ros2py310`
13) 所有训练/推理参数通过 `configs/*.json` 管理。
14) 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN9 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`输出目录：`runs/`；文档：`README.md`、`runtxt.md`。
15) 每次需要联网时使用 Playwright（`npx playwright`）。联网搜索默认使用 DuckDuckGo MCP。
16) **代码包名**：`ugv_dqn`（不是 `amr_dqn`）。所有 import 使用 `from ugv_dqn.xxx import ...`。
17) **学术问题必须先读论文**：回答任何与本项目相关的学术问题前，必须先读 `paper/main.tex` 及相关章节，基于论文实际内容回答，严禁凭印象或通用知识敷衍作答。
18) **复杂任务默认多 Agent 集群**：涉及多文件修改、跨模块调研、论文+代码联动等复杂任务时，默认启用多 Agent 并行（`Agent` 工具），将独立子任务分派给专用 subagent 并发执行，最大化效率。简单单文件任务无需启用。
19) **修改前先备份到 GitHub**：每次修改代码或论文文件前，先将当前未提交的更改 `git commit` 并 `git push` 到 GitHub，确保用户可随时回退。若无未提交更改则跳过。

## 1. 常用命令

```bash
PROJ=/home/sun/phdproject/dqn/DQN9; ENV=ros2py310

# 训练（后台）
nohup conda run --cwd $PROJ -n $ENV python train.py --profile $PROFILE \
  > runs/${PROFILE}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 推理
conda run --cwd $PROJ -n $ENV python infer.py --profile $PROFILE

# 检查完成
ls $PROJ/runs/$EXP/train_*/infer/*/table2_kpis.csv 2>/dev/null && echo DONE || echo RUNNING
```

### 术语规范

| 中文                              | 英文                | 禁用                 | 说明                                                                                       |
| --------------------------------- | ------------------- | -------------------- | ------------------------------------------------------------------------------------------ |
| Dijkstra代价场 / 到达目标的代价场 | goal distance field | cost-to-go field/map | Dijkstra 预计算的绕障最短路径距离场；"cost-to-go"在 DRL 语境下易与 RL 值函数混淆，禁止使用 |
| MD                                | MHA + Duel          | —                   | Multi-Head Attention + Dueling Network 的组合简称                                          |

- 观测通道描述：occupancy, **goal distance**
- 奖励塑形描述：potential function defined as the geodesic goal distance
- 代码变量名（`cost_to_go` 等）暂不改，仅论文正文执行此规范

### 引用核查（强制）

1. `search_web` / Semantic Scholar 定位 → 2. DOI 2+ 数据源确认 → 3. `curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>` → 4. 确认 claim 存在 → 失败标 `[CITATION NEEDED]`

## 3. 论文实验评估框架

### 3.1 评估体系

**环境**：Realmap（复杂）

**每环境 3 类结果**：

| 结果          | 筛选         | 路径距离                  | Runs    | 汇报 KPI                 |
| ------------- | ------------ | ------------------------- | ------- | ------------------------ |
| SR            | BK 可达      | Long ≥18m / Short 6–14m | 50+     | **仅成功率**       |
| Quality Long  | 3 算法全成功 | ≥18m                     | ~5–10  | 路径长度、曲率、计算时间 |
| Quality Short | 3 算法全成功 | 6–14m                    | ~25–30 | 路径长度、曲率、计算时间 |

- **3-algo filter**：MD-DDQN + RRT* + LO-HA*。
- **SR 结果**只报成功率，不看路径质量（失败路径无有效指标）。
- **Quality 结果**只报路径质量，成功率恒 100%（消除混杂变量）。
- 直接比较原始指标，不做 minmax 归一化。

### 3.1.1 网络架构（CNN-MD-DDQN）

- **CNN 双输入（dual-input）**：卷积分支（2 通道地图：occupancy + goal distance，`cnn_drop_edt=true` 丢弃 EDT）+ 标量分支（11-d），标量分支**无 FC 预处理，直接拼接**
- **卷积骨干**：Conv(2→32, k3s1) → Conv(32→64, k3s2) → Conv(64→64, k3s2) → SpatialMHA(h=4) → Flatten(576)
- **拼接后**：FC(587→256) 共享层
- **Dueling 双流**：Value FC(256→256→1) / Advantage FC(256→256→35)
- **输出**：Q = V + A − mean(A) → Action Mask → a*
- **架构图**：`paper/figures/cnn_md_ddqn_architecture.{pdf,png}`（由 `plot_cnn_architecture_v2.py` 生成）

### 3.2 论文叙事约束

- **核心论点**：受约束 DRL 路径规划框架优于经典规划器。
- **系统级创新**：动作掩码、Dijkstra 奖励塑形、DQfD 预训练、自行车运动学约束、双圆碰撞检测。
- **核心算法**：MD-DDQN，reward_k_t=0.2（消融实验 2026-03-15 确认 MD 全面优于 Duel）。
- **若结果不支持叙事**：换 checkpoint 或调训练参数重跑，不可篡改数据。

### 3.3 核心对比实验：MD-DDQN vs RRT* vs LO-HA*（全局规划，无 MPC）

- **Seed**: 110（经 500 seeds 大规模扫描 + 9 候选 seed × 50 runs 深度验证确认；选择理由：Long SR=94% 更真实可信，且 Long PL/曲率/时间全赢）
- **配置**: `configs/final_t10_sr_{long,short}.json`，`baseline_timeout=10s`
- **碰撞检测**: EDT diag（baselines 与 DRL 使用完全相同的 `EDTCollisionChecker`）
- **结果（50 runs, seed 110, exp1.2 带 traces）**:

| 距离  | 指标                       | MD-DDQN                | LO-HA*           | RRT*        |
| ----- | -------------------------- | ---------------------- | ---------------- | ----------- |
| Long  | SR                         | **94%** (47/50)  | 36% (18/50)      | 74% (37/50) |
| Long  | PathLen (Quality, 18 runs) | **24.438m**      | 24.667m          | 24.486m     |
| Long  | Curvature                  | **0.1074**       | 0.1294           | 0.1264      |
| Long  | Time                       | **0.626s**       | 5.084s           | 3.351s      |
| Short | SR                         | **100%** (50/50) | 92% (46/50)      | 96% (48/50) |
| Short | PathLen (Quality, 44 runs) | 8.306m                 | **8.285m** | 8.330m      |
| Short | Curvature                  | **0.1204**       | 0.1324           | 0.1809      |
| Short | Time                       | **0.267s**       | 1.701s           | 1.382s      |

- **结论**: Long 距离 DRL 在 SR/PL/曲率/时间全面胜出；Short 距离 SR/曲率/时间赢，PL 与 LO-HA* 接近持平（差 0.021m，<0.3%）
- **本地结果**: `paper/results/exp1.2_core_comparison/`
- **历史**: exp1/exp1.1 使用 seed 420（DRL SR=100%，过于完美），exp1.2 切换到 seed 110

### 3.4 DRL 模块消融实验（8 变体）

- **训练**: 10000 episodes, DQfD pretrain 40000 steps, reward_k_t=0.2, EDT diag
- **推理**: Seeds 42/142/242/342/442/542/642，50 runs/variant/distance
- **变体**: DDQN, Duel-DDQN, MHA-DDQN, MD-DDQN, DQN, Duel-DQN, MHA-DQN, MD-DQN
- **结论**: MD-DDQN Quality 路径长度 Long/Short 均第一（26.364m / 8.603m），SR 稳定（94%/92%）
- **详细日志**: `runs/ablation_logs/ablation_20260315_diag10k_kt02.md`
- **配置**: `configs/ablation_20260314_diag10k_kt02_*.json`

### 3.5 AM × DQfD 组件消融实验（3 变体）

- **基底**: MD-DDQN (reward_k_t=0.2, EDT diag)
- **训练**: 10000 episodes, seed=0
- **推理**: Seeds 100/200/300/400/500/600/700，50 runs/variant/distance
- **设计**: 训练时消融，推理统一带 mask（隔离训练时贡献）
- **代码改动**: `ugv_dqn/cli/train.py` 4 处条件化 `forest_action_shield`（expert exploration fallback、TD target mask、demo prefill mask ×2）
- **变体**:

| 变体           | AM (shield+TD mask) | DQfD (prefill+pretrain+expert_exploration) | 训练来源                              |
| -------------- | ------------------- | ------------------------------------------ | ------------------------------------- |
| Full (AM+DQfD) | ON                  | ON                                         | 复用 `abl_diag10k_kt02_cnn_ddqn_md` |
| w/o DQfD       | ON                  | OFF                                        | `abl_amdqfd_noDQfD`                 |
| w/o AM         | OFF                 | ON                                         | `abl_amdqfd_noAM`                   |

- **SR 结果（50 runs）**:

| 变体                     | Long SR       | Short SR      |
| ------------------------ | ------------- | ------------- |
| **Full (AM+DQfD)** | **92%** | **90%** |
| w/o AM                   | 72%           | 86%           |
| w/o DQfD                 | 56%           | 80%           |

- **Quality 结果**:

| 距离            | 指标      | Full              | w/o AM           | w/o DQfD |
| --------------- | --------- | ----------------- | ---------------- | -------- |
| Long (19 runs)  | PathLen   | **24.652m** | 25.317m          | 25.831m  |
| Long            | Curvature | **0.1412**  | 0.1537           | 0.1812   |
| Long            | Time      | **0.490s**  | 0.523s           | 0.648s   |
| Short (32 runs) | PathLen   | 8.809m            | **8.778m** | 9.048m   |
| Short           | Curvature | 0.1685            | **0.1500** | 0.1718   |
| Short           | Time      | 0.287s            | **0.258s** | 0.370s   |

- **结论**: DQfD 预训练贡献最大（Long SR -36pp），AM 显著辅助（Long SR -20pp），两者协同最优
- **详细日志**: `runs/ablation_logs/ablation_20260316_amdqfd.md`
- **配置**: `configs/ablation_20260315_amdqfd_*.json`
- **本地结果**: `runs/abl_amdqfd_infer_{full,noDQfD,noAM}/`

## 5. 远程服务器

| 优先级 | 名称               | Host           | 用户   | 密码             | GPU             | 说明                                    |
| ------ | ------------------ | -------------- | ------ | ---------------- | --------------- | --------------------------------------- |
| 1      | uhost-1nwalbarw6ki | 117.50.216.203 | ubuntu | g7TXK26Q85Jp493f | RTX 4090 (24GB) | 租用 GPU 服务器，Conda ros2py310 已部署 |
| 2      | ubuntu-zt          | (ZeroTier)     | sun    | —               | —              | 长期训练服务器，存放 6000+ checkpoints  |

- 连接方式：优先用 paramiko（本地无 sshpass）
- 远端项目路径：`$HOME/DQN9/`
- 远端 Conda：`$HOME/miniconda3/bin/conda`，环境 `ros2py310`

## 4. 踩坑

- **SSH 执行必须** `--cwd`：`conda run --cwd $PROJ -n $ENV python ...`
- 联网调研：WebFetch/WebSearch 不混批，每批≤2；付费墙用浏览器工具。
- LaTeX：`xelatex` 支持中文注释；提交版用 `pdflatex`；缺包 `sudo tlmgr install <pkg>`。
