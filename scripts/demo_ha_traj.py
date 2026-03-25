#!/usr/bin/env python3
"""Hybrid A* 21-bin 长距离轨迹（>22m）。"""

import sys, math, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ugv_dqn.maps import get_map_spec
from ugv_dqn.baselines.pathplan import (
    grid_map_from_obstacles, default_ackermann_params, forest_two_circle_footprint,
)
from ugv_dqn.third_party.pathplan.primitives import MotionPrimitive
from ugv_dqn.third_party.pathplan.hybrid_a_star.planner import HybridAStarPlanner
from ugv_dqn.third_party.pathplan.robot import AckermannState
from ugv_dqn.env import compute_edt_distance_m
from ugv_dqn.third_party.pathplan.geometry import EDTCollisionChecker
from ugv_dqn.metrics import avg_abs_curvature

# ── 地图与碰撞检测 ──
spec = get_map_spec("realmap_a")
cell_size_m = 0.1
grid = spec.grid_y0_bottom
grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=cell_size_m)
params = default_ackermann_params()
footprint = forest_two_circle_footprint()
_edt = compute_edt_distance_m(spec.obstacle_grid().astype(np.uint8), cell_size_m=cell_size_m)
checker = EDTCollisionChecker(edt_dist_m=_edt, cell_size_m=cell_size_m,
                              footprint=footprint, edt_collision_margin="diag")

def make_prims(n_bins, step_length=0.3):
    delta_max = params.max_steer
    bins = [delta_max * (-1.0 + 2.0 * i / (n_bins - 1)) for i in range(n_bins)]
    prims = []
    for d in bins:
        prims.append(MotionPrimitive(d, +1, step_length, weight=1.0))
    for d in bins:
        prims.append(MotionPrimitive(d, -1, step_length, weight=1.2))
    return prims

def run_ha(sx, sy, gx, gy, n_bins, timeout=30.0, max_nodes=500_000):
    prims = make_prims(n_bins)
    st = AckermannState(sx * cell_size_m, sy * cell_size_m, 0.0)
    gl = AckermannState(gx * cell_size_m, gy * cell_size_m, 0.0)
    planner = HybridAStarPlanner(
        grid_map, footprint, params,
        primitives=prims, collision_checker=checker,
        curvature_step=0.05, max_curvature_ratio=2.0,
        sigma1=0.4, sigma2=0.6,
        reeds_shepp_heuristic_max_dist=50.0,
        goal_theta_tol=math.pi,
    )
    t0 = time.perf_counter()
    path, stats = planner.plan(st, gl, timeout=timeout, max_nodes=max_nodes)
    dt = time.perf_counter() - t0
    trace = stats.get("trace_poses", [])
    if trace and len(trace) >= 2:
        pts_cell = [(x / cell_size_m, y / cell_size_m) for x, y, _th in trace]
        pts_m = [(x, y) for x, y, _th in trace]
    elif path:
        pts_cell = [(s.x / cell_size_m, s.y / cell_size_m) for s in path]
        pts_m = [(s.x, s.y) for s in path]
    else:
        pts_cell, pts_m = [], []
    return pts_cell, pts_m, dt, bool(path), stats

# ── 长距离 case: 横穿地图 (~22m+) ──
# realmap_a 是 410x129 cells = 41x12.9m
# 起点在左侧，终点在右侧中部
cases = [
    {"start": (34, 29), "goal": (250, 40), "label": "Long ~22m"},
    {"start": (34, 29), "goal": (350, 40), "label": "Long ~32m"},
]

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

for ax, case in zip(axes, cases):
    sx, sy = case["start"]
    gx, gy = case["goal"]

    n_bins = 21
    print(f"Running {case['label']} with {n_bins}-bin ...")
    pts_cell, pts_m, dt, ok, stats = run_ha(sx, sy, gx, gy, n_bins, timeout=30.0, max_nodes=500_000)

    ax.imshow(grid, origin="lower", cmap="gray_r", alpha=0.35)

    if ok and pts_m:
        curv = avg_abs_curvature(pts_m)
        plen = sum(math.hypot(pts_m[i+1][0]-pts_m[i][0], pts_m[i+1][1]-pts_m[i][1])
                   for i in range(len(pts_m)-1))
        expns = stats.get("expansions", 0)
        xs = [p[0] for p in pts_cell]
        ys = [p[1] for p in pts_cell]
        ax.plot(xs, ys, "b-", linewidth=2.0)
        ax.plot(xs, ys, "b.", markersize=1.5, alpha=0.5)
        info = f"21-bin: len={plen:.1f}m  κ={curv:.3f}/m  t={dt:.1f}s  expns={expns}  pts={len(pts_m)}"
        print(f"  OK: {info}")
    else:
        reason = stats.get("failure_reason", "?")
        expns = stats.get("expansions", 0)
        timed_out = stats.get("timed_out", False)
        info = f"21-bin: FAILED  t={dt:.1f}s  expns={expns}  timeout={timed_out}  reason={reason}"
        print(f"  {info}")

    ax.plot(sx, sy, "go", markersize=10, zorder=5)
    ax.plot(gx, gy, "r*", markersize=14, zorder=5)
    ax.set_title(f"{case['label']}\n{info}", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

plt.suptitle("Hybrid A* 21-bin (Dang 2022) — long distance paths", fontsize=13)
plt.tight_layout()
out = Path("runs/trajectory_figs/ha_dang2022_long.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
