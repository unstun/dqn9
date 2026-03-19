#!/usr/bin/env python3
"""生成观测空间示意图：原始占据图 → 12×12 降采样三通道。

用法：
    cd DQN9
    python paper/figures/plot_observation_channels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ugv_dqn.env import (
    _downsample_map_preserve_aspect,
    compute_edt_distance_m,
    dijkstra_cost_to_goal_m,
)
from ugv_dqn.maps import get_map_spec

# ── 参数 ──────────────────────────────────────────────────────────────
ENV_NAME = "realmap_a"
OBS_MAP_SIZE = 12
CELL_SIZE_M = 0.1
OD_CAP_M = 2.0  # EDT 截断距离
DPI = 300
OUT_DIR = Path(__file__).resolve().parent

# ── 加载地图 ──────────────────────────────────────────────────────────
spec = get_map_spec(ENV_NAME)
grid = spec.obstacle_grid()  # (H, W) uint8, 1=obstacle
H, W = grid.shape
start_xy = spec.start_xy
goal_xy = spec.goal_xy

print(f"地图: {ENV_NAME}, 尺寸: {W}×{H}, 起点: {start_xy}, 终点: {goal_xy}")

# ── 计算三个场 ────────────────────────────────────────────────────────
# 1) 占据图
occ = grid.astype(np.float32)

# 2) Dijkstra 目标距离场
traversable = (grid == 0).astype(np.uint8)
cost_to_goal = dijkstra_cost_to_goal_m(
    traversable, goal_xy=goal_xy, cell_size_m=CELL_SIZE_M
)
# 有限上界填充
cost_fill = np.nanmax(cost_to_goal[np.isfinite(cost_to_goal)]) * 1.2
cost_clamped = np.where(np.isfinite(cost_to_goal), cost_to_goal, cost_fill)
cost_norm = max(cost_fill, 1.0)
cost01 = np.clip(cost_clamped / cost_norm, 0.0, 1.0)

# 3) EDT 安全距离场
edt_m = compute_edt_distance_m(grid, cell_size_m=CELL_SIZE_M)
edt01 = np.clip(edt_m / OD_CAP_M, 0.0, 1.0)

# ── 降采样到 12×12 ───────────────────────────────────────────────────
occ_ds = _downsample_map_preserve_aspect(
    occ, OBS_MAP_SIZE, interpolation=cv2.INTER_NEAREST, pad_value=1.0
)
cost_ds = _downsample_map_preserve_aspect(
    cost01, OBS_MAP_SIZE, pad_value=1.0
)
edt_ds = _downsample_map_preserve_aspect(
    edt01, OBS_MAP_SIZE, pad_value=0.0
)

print(f"降采样: {W}×{H} → {OBS_MAP_SIZE}×{OBS_MAP_SIZE}")

# ── 配色 ─────────────────────────────────────────────────────────────
# 占据图：白=自由, 黑=障碍
cmap_occ = "gray_r"
# Dijkstra：深蓝(近) → 浅黄(远)
cmap_cost = "viridis"
# EDT：红(近障碍) → 绿(远障碍)
cmap_edt = "RdYlGn"

# ── 绘图 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))

# 标记起点终点的辅助函数
def mark_points(ax, sx, sy, gx, gy):
    ax.plot(sx, sy, "o", color="#2196F3", markersize=5, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    ax.plot(gx, gy, "*", color="#F44336", markersize=7, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)

def mark_points_ds(ax, sx, sy, gx, gy, orig_size, ds_size):
    """在降采样图上标记起点终点"""
    scale = ds_size / orig_size
    ax.plot(sx * scale, sy * scale, "o", color="#2196F3", markersize=6,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.plot(gx * scale, gy * scale, "*", color="#F44336", markersize=8,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)

# ── 第一行：原始分辨率 ──────────────────────────────────────────────
# (a) 占据图
ax = axes[0, 0]
ax.imshow(occ, origin="lower", cmap=cmap_occ, interpolation="nearest")
mark_points(ax, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
ax.set_title(f"(a) Occupancy ({W}×{H})", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# (b) Dijkstra 目标距离场
ax = axes[0, 1]
im_cost = ax.imshow(
    np.ma.masked_where(grid == 1, cost_to_goal),
    origin="lower", cmap=cmap_cost, interpolation="bilinear"
)
# 障碍物叠加为灰色
ax.imshow(
    np.ma.masked_where(grid == 0, grid.astype(float)),
    origin="lower", cmap="gray_r", alpha=0.7, interpolation="nearest"
)
mark_points(ax, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
ax.set_title(f"(b) Goal distance ({W}×{H})", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# (c) EDT 安全距离场
ax = axes[0, 2]
im_edt = ax.imshow(
    np.ma.masked_where(grid == 1, edt_m),
    origin="lower", cmap=cmap_edt, interpolation="bilinear",
    vmin=0, vmax=OD_CAP_M
)
ax.imshow(
    np.ma.masked_where(grid == 0, grid.astype(float)),
    origin="lower", cmap="gray_r", alpha=0.7, interpolation="nearest"
)
mark_points(ax, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
ax.set_title(f"(c) EDT clearance ({W}×{H})", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# ── 第二行：12×12 降采样 ────────────────────────────────────────────
N = OBS_MAP_SIZE

# (d) 占据图 12×12
ax = axes[1, 0]
ax.imshow(occ_ds, origin="lower", cmap=cmap_occ, interpolation="nearest",
          vmin=0, vmax=1)
mark_points_ds(ax, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1],
               max(H, W), N)
# 画网格线
for i in range(N + 1):
    ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
    ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
ax.set_title(f"(d) Occupancy ({N}×{N})", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# (e) Dijkstra 12×12
ax = axes[1, 1]
ax.imshow(cost_ds, origin="lower", cmap=cmap_cost, interpolation="nearest",
          vmin=0, vmax=1)
mark_points_ds(ax, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1],
               max(H, W), N)
for i in range(N + 1):
    ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
    ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
ax.set_title(f"(e) Goal distance ({N}×{N})", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# (f) EDT 12×12
ax = axes[1, 2]
ax.imshow(edt_ds, origin="lower", cmap=cmap_edt, interpolation="nearest",
          vmin=0, vmax=1)
mark_points_ds(ax, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1],
               max(H, W), N)
for i in range(N + 1):
    ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
    ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
ax.set_title(f"(f) EDT clearance ({N}×{N})", fontsize=9)
ax.set_xticks([])
ax.set_yticks([])

# ── 行标签 ───────────────────────────────────────────────────────────
fig.text(0.02, 0.72, "Original", fontsize=10, fontweight="bold",
         rotation=90, va="center")
fig.text(0.02, 0.30, "Downsampled", fontsize=10, fontweight="bold",
         rotation=90, va="center")

# ── 箭头连接上下行 ──────────────────────────────────────────────────
for col in range(3):
    fig.add_artist(plt.annotate(
        "", xy=(0.22 + col * 0.305, 0.46), xytext=(0.22 + col * 0.305, 0.50),
        xycoords="figure fraction", textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    ))

plt.tight_layout(rect=[0.04, 0.0, 1.0, 1.0])
plt.subplots_adjust(hspace=0.25)

# ── 保存 ─────────────────────────────────────────────────────────────
out_png = OUT_DIR / "observation_channels.png"
out_pdf = OUT_DIR / "observation_channels.pdf"
fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight", pad_inches=0.1)
fig.savefig(str(out_pdf), dpi=DPI, bbox_inches="tight", pad_inches=0.1)
print(f"已保存: {out_png}")
print(f"已保存: {out_pdf}")
plt.close(fig)
