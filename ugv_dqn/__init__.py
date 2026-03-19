"""ugv_dqn — 用于自主移动机器人路径规划的 DQN/DDQN 算法包。

包结构
------
agents.py          DQN/DDQN/PDDQN 智能体（TD 学习、目标网络、DQfD 专家损失）
env.py             Gymnasium 环境：UGVBicycleEnv（Ackermann 自行车运动学）
networks.py        Q 网络架构：MLPQNetwork、CNNQNetwork
replay_buffer.py   均匀经验回放缓冲区，支持 DQfD 演示数据保留
reward_norm.py     Welford 在线奖励归一化器（均值/标准差 + 裁剪）
forest_policy.py   统一的可行动作选择流程（训练与推理共享相同逻辑）
schedules.py       Epsilon 衰减策略（线性、自适应 sigmoid）
smoothing.py       Chaikin 角切割路径平滑器
metrics.py         路径 KPI 辅助函数：长度、曲率、转角
config_io.py       JSON 配置加载 + argparse 集成
runtime.py         PyTorch/CUDA/Matplotlib 后端设置
runs.py            带时间戳的实验运行目录管理

子包
----
cli/               入口点：train.py、infer.py、benchmark.py、config.py、precompute_forest_paths.py
maps/              地图定义：forest（程序化生成 A-D）、realmap（PGM）、预计算专家路径
baselines/         经典规划器：Hybrid A*、RRT*（对 third_party.pathplan 的封装）
third_party/       内置路径规划库：hybrid_a_star、rrt、几何工具
"""
