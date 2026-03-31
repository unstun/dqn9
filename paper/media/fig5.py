#!/usr/bin/env python3
"""生成 12×12 占据图 → Dijkstra 目标距离场 示意图（图5）。

手工构造一个简单的 12×12 地图，障碍物清晰可辨。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ugv_dqn.env import dijkstra_cost_to_goal_m

# ── 手工构造 12×12 占据图 ─────────────────────────────────────────────
# 0=free, 1=obstacle, y=0 在底部
grid = np.zeros((12, 12), dtype=np.uint8)

# L 形障碍物
grid[4:9, 3] = 1
grid[4, 3:7] = 1

# 右上角小块
grid[8:10, 8:10] = 1

# 底部墙
grid[1, 6:10] = 1

N = 12
CELL_SIZE = 1.0  # 每格 1m，方便展示

start_xy = (1, 1)
goal_xy = (10, 10)

# ── 计算场 ────────────────────────────────────────────────────────────
traversable = (grid == 0).astype(np.uint8)
cost = dijkstra_cost_to_goal_m(traversable, goal_xy=goal_xy, cell_size_m=CELL_SIZE)
cost_fill = np.nanmax(cost[np.isfinite(cost)]) * 1.05
cost_display = np.where(np.isfinite(cost), cost, cost_fill)

# ── 绘图 ─────────────────────────────────────────────────────────────
# gridspec_kw 给右图多留 colorbar 空间，但两个 Axes 本身等宽
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3),
                         gridspec_kw={"width_ratios": [1, 1]})

def draw_grid_lines(ax, n):
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.6)
        ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.6)

def mark_se(ax):
    ax.plot(*start_xy, "o", color="#2196F3", markersize=8,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax.plot(*goal_xy, "*", color="#F44336", markersize=12,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)

# (a) 占据图
ax = axes[0]
ax.imshow(grid, origin="lower", cmap="gray_r", interpolation="nearest",
          vmin=0, vmax=1)
draw_grid_lines(ax, N)
mark_se(ax)
ax.set_xlabel("(a) Occupancy grid", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# (b) Dijkstra 目标距离场
ax = axes[1]
masked_cost = np.ma.masked_where(grid == 1, cost_display)
im = ax.imshow(masked_cost, origin="lower", cmap="viridis", interpolation="nearest")
# 障碍物填灰
ax.imshow(np.ma.masked_where(grid == 0, grid.astype(float)),
          origin="lower", cmap="gray_r", interpolation="nearest", alpha=0.8)
draw_grid_lines(ax, N)
mark_se(ax)
ax.set_xlabel("(b) Goal distance field", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# 图例（右上角带框）
legend_elements = [
    mpatches.Patch(facecolor="black", edgecolor="gray", label="Obstacle"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="Start"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#F44336",
               markersize=10, label="Goal"),
]
axes[0].legend(handles=legend_elements, loc="upper left",
               fontsize=7, frameon=True, fancybox=False,
               edgecolor="gray", facecolor="white", framealpha=0.9)

plt.tight_layout()

OUT = Path(__file__).resolve().parent
fig.savefig(str(OUT / "fig5.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
fig.savefig(str(OUT / "fig5.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"已保存: {OUT / 'fig5.png'}")
plt.close(fig)
