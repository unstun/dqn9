# DQN9 项目规则（论文写作 + 工程开发）

> 作用域：`/home/sun/phdproject/dqn/DQN9/**`
> 继承自 DQN8，代码包已从 `amr_dqn` 重命名为 `ugv_dqn`。

## 0. 硬约束

1) 每次回复以"帅哥，"开头。
2) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
3) 默认中文回复；论文正文先中文写作，定稿后统一英文润色。README 等项目文档也用中文撰写。
4) **严禁凭记忆生成 BibTeX**——未核实引用标 `[CITATION NEEDED]`。
5) `CLAUDE.md` 为唯一项目规则文件。
6) 纯文档改动豁免 `configs/` 新增规则；代码改动须在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`。
7) **消融实验留档**：结束后在 `runs/ablation_logs/` 写 `ablation_YYYYMMDD_<topic>.md`。
8) 每次可以先尝试ssh再本地计算
9) **代码搜索策略**：语义理解/探索代码库时**始终优先使用 ACE**（`mcp__augment-context-engine__codebase-retrieval`）；精确匹配标识符/字符串时使用 `Grep`（rg）。禁止用 Bash 调 grep/rg。ACE 调用若报错/超时，立即回退到 `Grep` + `Glob` 继续工作，不阻塞流程。
10) Conda 环境：`ros2py310`
11) 所有训练/推理参数通过 `configs/*.json` 管理。
12) 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN9 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`输出目录：`runs/`；文档：`README.md`、`runtxt.md`。
13) 每次需要联网时使用 Playwright（`npx playwright`）。联网搜索默认使用 DuckDuckGo MCP。
14) **代码包名**：`ugv_dqn`（不是 `amr_dqn`）。所有 import 使用 `from ugv_dqn.xxx import ...`。

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

- 观测通道描述：occupancy, **goal distance**, EDT clearance
- 奖励塑形描述：potential function defined as the geodesic goal distance
- 代码变量名（`cost_to_go` 等）暂不改，仅论文正文执行此规范

### 引用核查（强制）

1. `search_web` / Semantic Scholar 定位 → 2. DOI 2+ 数据源确认 → 3. `curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>` → 4. 确认 claim 存在 → 失败标 `[CITATION NEEDED]`

## 3. 论文实验评估框架

### 3.1 评估体系

**环境**：Realmap（复杂）

**每环境 3 类结果**：

| 结果          | 筛选              | 路径距离                  | Runs    | 汇报 KPI                            |
| ------------- | ----------------- | ------------------------- | ------- | ----------------------------------- |
| SR            | BK 可达           | Long ≥18m / Short 6–14m | 50+     | **仅成功率**                  |
| Quality Long  | 3 算法全成功      | ≥18m                     | ~5–10  | 路径长度、曲率、计算时间 |
| Quality Short | 3 算法全成功      | 6–14m                    | ~25–30 | 路径长度、曲率、计算时间 |

- **3-algo filter**：CNN-DDQN+Duel + RRT* + LO-HA*。
- **SR 结果**只报成功率，不看路径质量（失败路径无有效指标）。
- **Quality 结果**只报路径质量，成功率恒 100%（消除混杂变量）。
- 直接比较原始指标，不做 minmax 归一化。

### 3.2 论文叙事约束

- **核心论点**：受约束 DRL 路径规划框架优于经典规划器。
- **系统级创新**：动作掩码、Dijkstra 奖励塑形、DQfD 预训练、自行车运动学约束、双圆碰撞检测。
- **核心算法**：CNN-DDQN+Duel，reward_k_t=0.2。
- **若结果不支持叙事**：换 checkpoint 或调训练参数重跑，不可篡改数据。

### 3.3 MPC 对比实验（有利 seeds）

- **Seeds**: 42, 142, 242, 342, 442, 542, 642（每 seed 7-8 runs，共 50）
- **配置**: `configs/reward_abl_mpc_kt02_sr_{long,short}.json`
- **结果**: DRL 全面优于 LO-HA* 和 RRT*（Long 6/6, Short 27/29 路径更短）

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
