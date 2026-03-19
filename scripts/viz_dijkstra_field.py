"""可视化 realmap 上的占据图、EDT 距离场、Dijkstra 目标距离场。"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ugv_dqn.maps import get_map_spec
from ugv_dqn.env import dijkstra_cost_to_goal_m, compute_edt_distance_m

# ---------- 加载 realmap ----------
spec = get_map_spec("realmap_a")
grid = spec.grid_y0_bottom          # 1=障碍, 0=自由
traversable = ~grid.astype(bool)    # True=可通行
goal_xy = (spec.goal_xy[0], spec.goal_xy[1])   # (371, 109)
start_xy = (spec.start_xy[0], spec.start_xy[1])
cell_size_m = 0.1

print(f"Map size: {grid.shape[1]}x{grid.shape[0]} cells "
      f"= {grid.shape[1]*cell_size_m:.1f}x{grid.shape[0]*cell_size_m:.1f} m")
print(f"Start: {start_xy}, Goal: {goal_xy}")

# ---------- 计算场 ----------
edt = compute_edt_distance_m(grid, cell_size_m=cell_size_m)
cost = dijkstra_cost_to_goal_m(traversable, goal_xy=goal_xy, cell_size_m=cell_size_m)

# ---------- 绘图 ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

# (a) 占据图
occ_cmap = ListedColormap(["white", "black"])
ax = axes[0]
ax.imshow(grid, origin="lower", cmap=occ_cmap, interpolation="nearest")
ax.plot(*start_xy, "go", markersize=8, label="Start")
ax.plot(*goal_xy, "r*", markersize=12, label="Goal")
ax.set_title("(a) Occupancy Grid", fontsize=13)
ax.legend(loc="upper left", fontsize=9)
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")

# (b) EDT 距离场
ax = axes[1]
im = ax.imshow(edt, origin="lower", cmap="viridis", interpolation="nearest")
ax.plot(*start_xy, "go", markersize=8)
ax.plot(*goal_xy, "r*", markersize=12)
ax.set_title("(b) EDT Clearance (m)", fontsize=13)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("distance to nearest obstacle (m)")
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")

# (c) Dijkstra 目标距离场
ax = axes[2]
cost_vis = cost.copy()
cost_vis[~np.isfinite(cost_vis)] = np.nan  # 不可达区域透明
im = ax.imshow(cost_vis, origin="lower", cmap="inferno_r", interpolation="nearest")
ax.plot(*start_xy, "go", markersize=8, label="Start")
ax.plot(*goal_xy, "r*", markersize=12, label="Goal")
ax.set_title("(c) Dijkstra Goal Distance Field (m)", fontsize=13)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("geodesic distance to goal (m)")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "runs", "viz_dijkstra_field.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
