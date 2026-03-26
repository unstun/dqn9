"""轨迹可视化：Short run0 (三算法全成功) + Short run2 (三算法全成功)"""
import pickle, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
ALGO_NAMES = ["CNN-DDQN+Duel", "Hybrid A*", "RRT*"]
LABELS = {"CNN-DDQN+Duel": "MD-DDQN (Ours)",
          "Hybrid A*": "Hybrid A* [Dang 2022]",
          "RRT*": "SS-RRT* [Yoon 2018]"}
COLORS = {"CNN-DDQN+Duel": "#1f77b4", "Hybrid A*": "#ff7f0e", "RRT*": "#2ca02c"}
LW     = {"CNN-DDQN+Duel": 2.5, "Hybrid A*": 1.8, "RRT*": 1.8}
LS     = {"CNN-DDQN+Duel": "-", "Hybrid A*": "--", "RRT*": "-."}
ZORDER = {"CNN-DDQN+Duel": 5, "Hybrid A*": 4, "RRT*": 3}


def plot_run(ax, data, run_idx, cell_size, title, algos=None):
    ogrid = data["obstacle_grid"]  # (H, W)
    paths = data["paths"]
    if algos is None:
        algos = ALGO_NAMES

    # 先收集所有轨迹坐标确定裁剪范围
    all_xy = []
    for algo in algos:
        k = ("realmap_a", run_idx, algo)
        if k in paths and paths[k]["success"]:
            xy = np.array(paths[k]["xy_cells"]) * cell_size
            all_xy.append(xy)
    if not all_xy:
        ax.set_title(f"{title} — no data"); return
    all_pts = np.concatenate(all_xy)
    pad = 3.0
    xmin, xmax = all_pts[:,0].min()-pad, all_pts[:,0].max()+pad
    ymin, ymax = all_pts[:,1].min()-pad, all_pts[:,1].max()+pad

    # 障碍物底图  ogrid shape=(rows=Y, cols=X)
    ny, nx = ogrid.shape
    ax.imshow(ogrid, origin="lower", cmap="Greys", alpha=0.4,
              extent=[0, nx*cell_size, 0, ny*cell_size])

    # 画轨迹
    for algo in algos:
        k = ("realmap_a", run_idx, algo)
        if k not in paths or not paths[k]["success"]:
            continue
        xy = np.array(paths[k]["xy_cells"]) * cell_size
        ax.plot(xy[:, 0], xy[:, 1],
                color=COLORS[algo], linewidth=LW[algo], linestyle=LS[algo],
                label=LABELS[algo], zorder=ZORDER[algo], alpha=0.9)

    # 起点终点（取 DRL 的）
    drl_k = ("realmap_a", run_idx, "CNN-DDQN+Duel")
    xy0 = np.array(paths[drl_k]["xy_cells"]) * cell_size
    ax.plot(*xy0[0],  "k*", markersize=16, zorder=10)
    ax.plot(*xy0[-1], "r*", markersize=16, zorder=10)
    ax.annotate("Start", xy0[0], fontsize=10, fontweight="bold",
                xytext=(5, 8), textcoords="offset points")
    ax.annotate("Goal",  xy0[-1], fontsize=10, fontweight="bold", color="red",
                xytext=(5, 8), textcoords="offset points")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.15)


# ── 加载 ──
data_short = pickle.load(open(HERE / "short" / "paths_for_plot.pkl", "rb"))
data_long  = pickle.load(open(HERE / "long"  / "paths_for_plot.pkl", "rb"))
cs = data_short["cell_size_m"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Short run 0: 三算法全成功
plot_run(axes[0], data_short, 0, cs, "Short Distance — Run #0 (all 3 succeed)")

# Short run 2: 三算法全成功
plot_run(axes[1], data_short, 2, cs, "Short Distance — Run #2 (all 3 succeed)")

fig.suptitle("g1t03 Trajectory Comparison — Pure Planning, goal_tol = 0.3 m",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "g1t03_traj_comparison.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
