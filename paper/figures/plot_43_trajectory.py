"""4.3 核心对比 -- 轨迹对比图 (1x2): Long / Short.

数据源: runs202643/infer/core_baseline_dqn_sr_{long,short}/*/paths_all.csv + map_meta.pkl
输出:   paper/figures/fig_43_trajectory.{pdf,png}
"""

from __future__ import annotations

import glob
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from style import apply_style, save_fig, RUNS3, C_MDDDQN, C_HASTAR, C_RRTSTAR

apply_style()

# ── 算法元数据 ────────────────────────────────────────────────────────
ALGO_NAMES = ["CNN-DQN+Duel", "Hybrid A*", "RRT*"]
LABELS = {
    "CNN-DQN+Duel": "MD-DQN (Ours)",
    "Hybrid A*":    "Improved HA*",
    "RRT*":         "Spline-RRT*",
}
COLORS = {
    "CNN-DQN+Duel": C_MDDDQN,
    "Hybrid A*":    C_HASTAR,
    "RRT*":         C_RRTSTAR,
}
LW     = {"CNN-DQN+Duel": 2.5, "Hybrid A*": 1.8, "RRT*": 1.8}
LS     = {"CNN-DQN+Duel": "-",  "Hybrid A*": "--", "RRT*": "-."}
ZORDER = {"CNN-DQN+Duel": 5,   "Hybrid A*": 4,    "RRT*": 3}


def load_data(dist: str):
    """从 runs202643 加载轨迹 CSV 和地图元数据。"""
    base = RUNS3 / "infer" / f"core_baseline_dqn_sr_{dist}"
    subdirs = sorted(base.iterdir())
    subdir = subdirs[-1]  # 取最新时间戳

    # 地图
    meta = pickle.load(open(subdir / "map_meta.pkl", "rb"))
    obstacle_grid = meta["obstacle_grid"]
    cell_size = meta["cell_size_m"]

    # 轨迹 CSV: env,run_idx,algo,point_idx,x_m,y_m,success
    df = pd.read_csv(subdir / "paths_all.csv")

    # 组装为 {(run_idx, algo): {"success": bool, "xy_m": np.ndarray}}
    paths = {}
    for (rid, algo), grp in df.groupby(["run_idx", "algo"]):
        grp = grp.sort_values("point_idx")
        success = bool(grp["success"].iloc[0])
        xy = grp[["x_m", "y_m"]].values
        paths[(int(rid), algo)] = {"success": success, "xy_m": xy}

    return obstacle_grid, cell_size, paths


def find_best_run(paths: dict, max_run: int = 50) -> int:
    """返回三算法全成功的 run_idx，优先选择路径长度适中的。"""
    for rid in range(max_run):
        if all(
            (rid, a) in paths and paths[(rid, a)]["success"]
            for a in ALGO_NAMES
        ):
            return rid
    # fallback: 最多成功的
    best_rid, best_cnt = 0, -1
    for rid in range(max_run):
        cnt = sum(1 for a in ALGO_NAMES
                  if (rid, a) in paths and paths[(rid, a)]["success"])
        if cnt > best_cnt:
            best_cnt = cnt
            best_rid = rid
    return best_rid


def plot_run(ax, obstacle_grid, cell_size, paths, run_idx, title):
    """在 ax 上绘制单个 run 的三算法轨迹。"""
    ny, nx = obstacle_grid.shape

    # 收集所有轨迹坐标，确定裁剪范围
    all_xy = []
    for algo in ALGO_NAMES:
        k = (run_idx, algo)
        if k in paths and paths[k]["success"]:
            all_xy.append(paths[k]["xy_m"])
    if not all_xy:
        ax.set_title(f"{title} (no data)")
        return
    all_pts = np.concatenate(all_xy)
    pad = 3.0
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

    # 障碍物底图
    ax.imshow(obstacle_grid, origin="lower", cmap="Greys", alpha=0.35,
              extent=[0, nx * cell_size, 0, ny * cell_size])

    # 轨迹
    for algo in ALGO_NAMES:
        k = (run_idx, algo)
        if k not in paths or not paths[k]["success"]:
            continue
        xy = paths[k]["xy_m"]
        ax.plot(xy[:, 0], xy[:, 1],
                color=COLORS[algo], linewidth=LW[algo], linestyle=LS[algo],
                label=LABELS[algo], zorder=ZORDER[algo], alpha=0.9)

    # 起点 / 终点标记
    ref_xy = None
    for algo in ALGO_NAMES:
        k = (run_idx, algo)
        if k in paths and paths[k]["success"]:
            ref_xy = paths[k]["xy_m"]
            break
    if ref_xy is not None:
        ax.plot(*ref_xy[0],  "k*", markersize=14, zorder=10, label="Start")
        ax.plot(*ref_xy[-1], "r*", markersize=14, zorder=10, label="Goal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.12, linewidth=0.5)

    scalebar = AnchoredSizeBar(
        ax.transData, 5.0, "5 m", loc="lower right",
        pad=0.4, borderpad=0.5, sep=4,
        frameon=True, size_vertical=0.15,
        fontproperties=fm.FontProperties(size=8),
    )
    ax.add_artist(scalebar)


def main():
    og_long, cs_long, paths_long = load_data("long")
    og_short, cs_short, paths_short = load_data("short")

    rid_long  = find_best_run(paths_long)
    rid_short = find_best_run(paths_short)
    print(f"Selected runs -- long: {rid_long}, short: {rid_short}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 3.5))

    plot_run(ax_a, og_long,  cs_long,  paths_long,  rid_long,  "(a) Long distance")
    plot_run(ax_b, og_short, cs_short, paths_short, rid_short, "(b) Short distance")

    # 统一图例
    all_handles, all_labels = [], []
    for ax in (ax_a, ax_b):
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    fig.legend(unique_h, unique_l,
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, "fig_43_trajectory")
    plt.close(fig)


if __name__ == "__main__":
    main()
