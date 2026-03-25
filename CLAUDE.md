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
5) 默认中文回复；论文正文先中文写作，定稿后统一英文润色。README 等项目文档也用中文撰写。论文中禁止使用括号补充说明，改用"即""由…构成""如图…所示"等行文方式替代；缩写定义除外，如"深度强化学习（DRL）"。公式中使用专业数学符号，禁止出现代码风格变量名如 `grid_size`、`cell_size`，应使用 $\Delta c$、$\delta$、$\epsilon$ 等标准记法。
6) **严禁凭记忆生成 BibTeX**——未核实引用标 `[CITATION NEEDED]`。
7) `CLAUDE.md` 为唯一项目规则文件。
8) 纯文档改动豁免 `configs/` 新增规则；代码改动须在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`。
9) **消融实验留档**：结束后在 `runs/ablation_logs/` 写 `ablation_YYYYMMDD_<topic>.md`。
10) 每次可以先尝试ssh再本地计算
11) **代码搜索策略**：语义理解/探索代码库时**始终优先使用 ACE**（`mcp__augment-context-engine__codebase-retrieval`）；精确匹配标识符/字符串时使用 `Grep`（rg）。禁止用 Bash 调 grep/rg。ACE 调用若报错/超时，立即回退到 `Grep` + `Glob` 继续工作，不阻塞流程。
12) Conda 环境：`ros2py310`
13) 所有训练/推理参数通过 `configs/*.json` 管理。
14) 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN9 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`输出目录：`runs/`；文档：`README.md`、`runtxt.md`。
15) 每次需要联网时使用 Playwright MCP 工具。DuckDuckGo MCP 已弃用，不再使用。
16) **代码包名**：`ugv_dqn`（不是 `amr_dqn`）。所有 import 使用 `from ugv_dqn.xxx import ...`。
17) **学术问题必须先读论文**：回答任何与本项目相关的学术问题前，必须先读 `paper/main.tex` 及相关章节，基于论文实际内容回答，严禁凭印象或通用知识敷衍作答。
18) **复杂任务默认多 Agent 集群**：涉及多文件修改、跨模块调研、论文+代码联动等复杂任务时，默认启用多 Agent 并行（`Agent` 工具），将独立子任务分派给专用 subagent 并发执行，最大化效率。简单单文件任务无需启用。
19) **修改前必须先备份到 GitHub（强制，无例外）**：每次修改代码或论文文件前，**必须先执行 `git add` + `git commit` + `git push`**，将当前所有未提交的更改推送到 GitHub，确保用户可随时回退。**严禁在未 push 的情况下开始任何文件修改。** 若无未提交更改则跳过。违反此规则等同于数据丢失风险。
20) **专业问题必须验证**：回答任何专业问题（学术规范、技术细节、算法原理等）前，必须通过联网搜索或本地文件验证，禁止凭 AI 记忆直接作答。不确定的内容标注不确定。
21) **远端训练前必须完整同步代码（强制，无例外）**：在远程服务器启动任何训练/推理前，**必须先执行 `rsync` 将本地完整项目（含 `configs/`、`ugv_dqn/`、`scripts/` 等）同步到远端**，确保远端代码与本地一致。**严禁在未同步的情况下启动远端训练。** 违反此规则可能导致整批训练作废。

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

| 中文                | 英文              | 禁用                             | 说明                                                                                                                 |
| ------------------- | ----------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Dijkstra 目标距离图 | goal distance map | cost-to-go field/map, 目标距离场 | Dijkstra 预计算的绕障最短路径距离图；"cost-to-go"在 DRL 语境下易与 RL 值函数混淆，禁止使用；"场"改"图"以匹配论文用语 |
| MD                  | MHA + Duel        | —                               | Multi-Head Attention + Dueling Network 的组合简称                                                                    |

- 观测通道描述：occupancy, **goal distance**
- 奖励塑形描述：potential function defined as the geodesic goal distance
- 代码变量名（`cost_to_go` 等）暂不改，仅论文正文执行此规范

### 引用核查（强制）

1. `search_web` / Semantic Scholar 定位 → 2. DOI 2+ 数据源确认 → 3. `curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>` → 4. 确认 claim 存在 → 失败标 `[CITATION NEEDED]`

## 3. 论文实验评估框架

> 详见 [`docs/experiment_framework.md`](docs/experiment_framework.md)（评估体系、网络架构、叙事约束、核心对比实验 3.3、消融实验 3.4/3.5 的完整数据表与结论）。写论文或分析实验数据时读取该文件。

## 5. 远程服务器

| 优先级 | 名称               | Host           | 用户   | 密码             | GPU             | 说明                                    |
| ------ | ------------------ | -------------- | ------ | ---------------- | --------------- | --------------------------------------- |
| 1      | uhost-1nwalbarw6ki | 117.50.216.203 | ubuntu | g7TXK26Q85Jp493f | RTX 4090 (24GB) | 租用 GPU 服务器，Conda ros2py310 已部署 |
| 2      | ubuntu-zt          | (ZeroTier)     | sun    | —               | —              | 长期训练服务器，存放 6000+ checkpoints  |

- 连接方式：优先用 paramiko（本地无 sshpass）
- 远端项目路径：`$HOME/DQN9/`
- 远端 Conda：`$HOME/miniconda3/bin/conda`，环境 `ros2py310`

## 4. 远端实验轮次识别

远端 `runs/` 目录中同一变体可能存在多轮推理结果，**通过时间戳区分 goal_tolerance 版本**：

| 训练时间戳 | 推理时间戳 | goal_tolerance | 说明 |
|-----------|-----------|:--------------:|------|
| `train_20260325_01xxxx` | `20260325_08xxxx` | **1.0m** | 旧容差，rsync 前的配置 |
| `train_20260325_12xxxx` | `20260325_18xxxx` | **0.3m** | 新容差，文献对齐 |

- 以 `20260325_18xxxx` 时间戳的推理结果为 0.3m 版本
- `.1` 后缀目录为旧版（goal_tolerance=1.0m 时期）的推理结果

## 5. 踩坑

- **SSH 执行必须** `--cwd`：`conda run --cwd $PROJ -n $ENV python ...`
- LaTeX：`xelatex` 支持中文注释；提交版用 `pdflatex`；缺包 `sudo tlmgr install <pkg>`。
- **论文写作规范**：写论文前必须遵守其中的术语、行文、段落规范。禁止捏造术语、过度包装简单概念、使用推销性语言。
