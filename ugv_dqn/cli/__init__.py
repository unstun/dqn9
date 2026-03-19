"""命令行入口点。

模块
-------
train.py                 主训练循环（1500+ 行）。
                         支持多环境、多算法、DQfD demo 预填充、
                         Hybrid A* 专家混合、奖励归一化、周期性评估。
                         入口：main() -> build_parser() + train_one()。

infer.py                 推理与评估（2700+ 行）。
                         加载训练好的模型，执行 rollout，计算 KPI，
                         与 Hybrid A* / RRT* baseline 比较，生成
                         CSV 表格 + 路径/控制图。
                         入口：main() -> build_parser()。

benchmark.py             编排 train -> infer 流水线并验证结果。
config.py                生成合并的 train+infer JSON 配置模板。
precompute_forest_paths.py  离线 Hybrid A* 专家路径缓存（用于 forest 地图）。
"""
