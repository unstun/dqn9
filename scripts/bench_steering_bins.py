#!/usr/bin/env python3
"""Benchmark: 转向角数量 vs 路径质量 vs 搜索时间。

使用与 infer.py 相同的 EDT 碰撞检测器和地图设置。
"""

import sys, math, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ugv_dqn.maps import get_map_spec
from ugv_dqn.baselines.pathplan import (
    grid_map_from_obstacles, default_ackermann_params, forest_two_circle_footprint,
    plan_hybrid_astar,
)
from ugv_dqn.third_party.pathplan.primitives import MotionPrimitive, default_primitives
from ugv_dqn.third_party.pathplan.hybrid_a_star.planner import HybridAStarPlanner
from ugv_dqn.third_party.pathplan.robot import AckermannState
from ugv_dqn.metrics import avg_abs_curvature

# ── 地图 ──
spec = get_map_spec("realmap_a")
cell_size_m = 0.1
grid_map = grid_map_from_obstacles(grid_y0_bottom=spec.grid_y0_bottom, cell_size_m=cell_size_m)
params = default_ackermann_params()
footprint = forest_two_circle_footprint()

# ── 构建 EDT 碰撞检测器 (与 infer.py 一致) ──
from ugv_dqn.env import compute_edt_distance_m
from ugv_dqn.third_party.pathplan.geometry import EDTCollisionChecker
_edt_dist_m = compute_edt_distance_m(spec.obstacle_grid().astype(np.uint8), cell_size_m=cell_size_m)
edt_checker = EDTCollisionChecker(
    edt_dist_m=_edt_dist_m, cell_size_m=cell_size_m,
    footprint=footprint, edt_collision_margin="diag",
)

sx, sy = 34, 29
gx, gy = 90, 40

def make_prims(n_bins: int, step_length: float = 0.3):
    """生成 n_bins 个均匀转向角 × {forward, reverse}。"""
    delta_max = params.max_steer
    if n_bins == 1:
        bins = [0.0]
    elif n_bins % 2 == 1:
        # 奇数: 保证包含 0
        half = n_bins // 2
        bins = [-delta_max + delta_max * i / half for i in range(half)]
        bins.append(0.0)
        bins.extend([delta_max * i / half for i in range(1, half + 1)])
    else:
        bins = [delta_max * (-1.0 + 2.0 * i / (n_bins - 1)) for i in range(n_bins)]
    prims = []
    for d in bins:
        prims.append(MotionPrimitive(d, +1, step_length, weight=1.0))
    for d in bins:
        prims.append(MotionPrimitive(d, -1, step_length, weight=1.2))
    return prims

print(f"Start: ({sx},{sy}) -> Goal: ({gx},{gy})")
print(f"{'bins':>5s}  {'prims':>5s}  {'time_s':>7s}  {'ok':>4s}  {'pts':>5s}  {'len_m':>6s}  {'curv':>6s}  {'expns':>7s}")
print("-" * 62)

for n_bins in [3, 5, 7, 9, 11, 15, 21]:
    prims = make_prims(n_bins)

    st = AckermannState(sx * cell_size_m, sy * cell_size_m, 0.0)
    gl = AckermannState(gx * cell_size_m, gy * cell_size_m, 0.0)

    planner = HybridAStarPlanner(
        grid_map, footprint, params,
        primitives=prims,
        collision_checker=edt_checker,
        curvature_step=0.05,
        max_curvature_ratio=2.0,
        sigma1=0.4, sigma2=0.6,
        reeds_shepp_heuristic_max_dist=50.0,
        goal_theta_tol=math.pi,
    )

    t0 = time.perf_counter()
    path, stats = planner.plan(st, gl, timeout=15.0, max_nodes=200_000)
    dt = time.perf_counter() - t0

    ok = bool(path) and not stats.get("timed_out", False)
    expns = stats.get("expansions", 0)
    if ok:
        trace = stats.get("trace_poses", [])
        if trace and len(trace) >= 2:
            pts_m = [(x, y) for x, y, _th in trace]
        else:
            pts_m = [(s.x, s.y) for s in path]
        curv = avg_abs_curvature(pts_m)
        plen = sum(math.hypot(pts_m[i+1][0]-pts_m[i][0], pts_m[i+1][1]-pts_m[i][1]) for i in range(len(pts_m)-1))
        print(f"{n_bins:5d}  {len(prims):5d}  {dt:7.2f}  {'OK':>4s}  {len(pts_m):5d}  {plen:6.2f}  {curv:6.3f}  {expns:7d}")
    else:
        reason = stats.get("failure_reason", "?")
        print(f"{n_bins:5d}  {len(prims):5d}  {dt:7.2f}  {'FAIL':>4s}  {'--':>5s}  {'--':>6s}  {'--':>6s}  {expns:7d}  {reason}")
