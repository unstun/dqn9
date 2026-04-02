"""4.5 AM x DQfD 组件消融 -- 轨迹对比图。

数据源: runs202642/infer/abl_amdqfd_*/*/paths_all.csv + map_meta.pkl
输出:   paper/figures/fig_45_trajectory.{pdf,png}

仅近距离实验，单幅图。
"""

from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from style import apply_style, save_fig, RUNS2, C_FULL, C_NOAM, C_NODQFD

apply_style()

# ── 变体元数据 ──────────────────────────────────────────────────────────
VARIANT_NAMES = ["Full (AM+DQfD)", "w/o AM", "w/o DQfD"]
COLORS  = {"Full (AM+DQfD)": C_FULL, "w/o AM": C_NOAM, "w/o DQfD": C_NODQFD}
LW      = {"Full (AM+DQfD)": 2.5,    "w/o AM": 1.8,    "w/o DQfD": 1.8}
LS      = {"Full (AM+DQfD)": "-",     "w/o AM": "--",   "w/o DQfD": "-."}
ZORDER  = {"Full (AM+DQfD)": 5,       "w/o AM": 4,      "w/o DQfD": 3}

DIR_TO_VARIANT = {
    "abl_amdqfd_full":   "Full (AM+DQfD)",
    "abl_amdqfd_noAM":  "w/o AM",
    "abl_amdqfd_noDQfD": "w/o DQfD",
}

INFER_DIR = RUNS2 / "infer"


def load_all_variants():
    """加载三个变体的轨迹数据，共享同一张地图。"""
    obstacle_grid = None
    cell_size = None
    # paths[variant][(run_idx)] = {"success": bool, "xy_m": np.ndarray}
    all_paths = {}

    for exp_dir in sorted(INFER_DIR.iterdir()):
        name = exp_dir.name
        if name not in DIR_TO_VARIANT:
            continue
        variant = DIR_TO_VARIANT[name]
        subdirs = sorted(exp_dir.iterdir())
        subdir = subdirs[-1]

        # 地图 (只读一次)
        if obstacle_grid is None:
            meta = pickle.load(open(subdir / "map_meta.pkl", "rb"))
            obstacle_grid = meta["obstacle_grid"]
            cell_size = meta["cell_size_m"]

        # 轨迹
        csv_path = subdir / "paths_all.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing paths_all.csv for {name}")
            continue
        df = pd.read_csv(csv_path)
        paths = {}
        for (rid, algo), grp in df.groupby(["run_idx", "algo"]):
            grp = grp.sort_values("point_idx")
            success = bool(grp["success"].iloc[0])
            xy = grp[["x_m", "y_m"]].values
            paths[int(rid)] = {"success": success, "xy_m": xy}
        all_paths[variant] = paths

    return obstacle_grid, cell_size, all_paths


def find_all_success_run(all_paths, max_run=50):
    """返回第一个三变体均成功的 run_idx。"""
    for rid in range(max_run):
        if all(
            rid in all_paths.get(v, {}) and all_paths[v][rid]["success"]
            for v in VARIANT_NAMES
        ):
            return rid
    return None


def main():
    obstacle_grid, cell_size, all_paths = load_all_variants()

    if obstacle_grid is None:
        print("[plot_45_trajectory] No data found, skipping.")
        return

    rid = find_all_success_run(all_paths)
    if rid is None:
        print("[plot_45_trajectory] No all-success run found, skipping.")
        return
    print(f"Selected run: {rid}")

    ny, nx = obstacle_grid.shape

    fig, ax = plt.subplots(figsize=(5, 4))

    # 收集裁剪范围
    all_xy = []
    for v in VARIANT_NAMES:
        if rid in all_paths.get(v, {}) and all_paths[v][rid]["success"]:
            all_xy.append(all_paths[v][rid]["xy_m"])
    all_pts = np.concatenate(all_xy)
    pad = 3.0
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

    # 障碍物底图
    ax.imshow(obstacle_grid, origin="lower", cmap="Greys", alpha=0.35,
              extent=[0, nx * cell_size, 0, ny * cell_size])

    # 轨迹
    for v in VARIANT_NAMES:
        if rid not in all_paths.get(v, {}) or not all_paths[v][rid]["success"]:
            continue
        xy = all_paths[v][rid]["xy_m"]
        ax.plot(xy[:, 0], xy[:, 1],
                color=COLORS[v], linewidth=LW[v], linestyle=LS[v],
                label=v, zorder=ZORDER[v], alpha=0.9)

    # 起终点
    ref_xy = all_paths["Full (AM+DQfD)"][rid]["xy_m"]
    ax.plot(*ref_xy[0],  "k*", markersize=14, zorder=10, label="Start")
    ax.plot(*ref_xy[-1], "r*", markersize=14, zorder=10, label="Goal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
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

    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig_45_trajectory")
    plt.close(fig)


if __name__ == "__main__":
    main()
