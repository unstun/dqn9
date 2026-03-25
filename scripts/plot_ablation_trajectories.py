#!/usr/bin/env python3
"""对比不同架构变体在栅格地图上的轨迹（goal_tolerance=0.3m 重训结果）。

选取所有 6 变体均成功的 run，在同一张占据栅格上叠加绘制轨迹。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── 项目路径 ─────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parents[1]
TS = "20260325_235700"  # 带 trace 的 0.3m 推理

# ── 变体定义 ─────────────────────────────────────────────────────────
VARIANTS = {
    "CNN-DDQN": {
        "dir": "abl_diag10k_kt02_infer_cnn_ddqn",
        "algo_file": "CNN-DDQN",
        "color": "#1f77b4",
        "ls": "-",
        "lw": 2.2,
    },
    "CNN-DDQN+Duel": {
        "dir": "abl_diag10k_kt02_infer_cnn_ddqn_duel",
        "algo_file": "CNN-DDQN_Duel",
        "color": "#ff7f0e",
        "ls": "-",
        "lw": 2.0,
    },
    "CNN-DDQN+MD": {
        "dir": "abl_diag10k_kt02_infer_cnn_ddqn_md",
        "algo_file": "CNN-DDQN_Duel",
        "color": "#d62728",
        "ls": "-",
        "lw": 2.2,
    },
    "CNN-DDQN+MHA": {
        "dir": "abl_diag10k_kt02_infer_cnn_ddqn_mha",
        "algo_file": "CNN-DDQN",
        "color": "#2ca02c",
        "ls": "--",
        "lw": 2.0,
    },
    "CNN-DQN": {
        "dir": "abl_diag10k_kt02_infer_cnn_dqn",
        "algo_file": "CNN-DQN",
        "color": "#9467bd",
        "ls": ":",
        "lw": 2.2,
    },
    "Scalar-only": {
        "dir": "abl_scalar_only_infer",
        "algo_file": "MLP-DDQN",
        "color": "#8c564b",
        "ls": "-.",
        "lw": 2.0,
    },
}

# ── 选取的 run index ─────────────────────────────────────────────────
RUN_INDICES = [3, 5]

# ── 加载地图（从任一变体） ────────────────────────────────────────────
first_dir = list(VARIANTS.values())[0]["dir"]
map_npz = PROJ / "runs" / first_dir / TS / "maps" / "realmap_a__grid_y0_bottom.npz"
npz = np.load(map_npz)
grid = npz["obstacle_grid"]
cell_size = float(npz["cell_size_m"])
H, W = grid.shape
extent_m = [0, W * cell_size, 0, H * cell_size]

# ── 绘图样式 ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
})
cmap_grid = mcolors.ListedColormap(["#fafafa", "#808080"])

# ── 逐 run 绘图 ─────────────────────────────────────────────────────
for run_idx in RUN_INDICES:
    traces = {}
    start_m = goal_m = None

    for label, cfg in VARIANTS.items():
        trace_dir = PROJ / "runs" / cfg["dir"] / TS / "traces"
        csv_path = trace_dir / f"realmap_a__{cfg['algo_file']}__run{run_idx}.csv"
        json_path = trace_dir / f"realmap_a__{cfg['algo_file']}__run{run_idx}.json"

        if not csv_path.exists():
            print(f"  SKIP {label} run{run_idx}: file not found")
            continue

        df = pd.read_csv(csv_path)
        meta = json.load(open(json_path))
        traces[label] = {"x": df["x_m"].values, "y": df["y_m"].values}

        if start_m is None:
            start_m = meta["start_m"]
            goal_m = meta["goal_m"]

    if not traces:
        continue

    # ── 裁剪区域 ──
    all_x = np.concatenate([t["x"] for t in traces.values()])
    all_y = np.concatenate([t["y"] for t in traces.values()])
    PAD = 2.0
    x_min = max(0, min(all_x.min(), start_m[0], goal_m[0]) - PAD)
    x_max = min(W * cell_size, max(all_x.max(), start_m[0], goal_m[0]) + PAD)
    y_min = max(0, min(all_y.min(), start_m[1], goal_m[1]) - PAD)
    y_max = min(H * cell_size, max(all_y.max(), start_m[1], goal_m[1]) + PAD)

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    fig_w = 12
    fig_h = max(fig_w * (crop_h / crop_w), 3.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 栅格底图
    ax.imshow(grid, origin="lower", extent=extent_m, cmap=cmap_grid,
              aspect="equal", interpolation="nearest", zorder=0)

    # 轨迹
    for label, cfg in VARIANTS.items():
        if label not in traces:
            continue
        t = traces[label]
        ax.plot(t["x"], t["y"], color=cfg["color"], linestyle=cfg["ls"],
                linewidth=cfg["lw"], label=label, zorder=2, alpha=0.85)

    # 起点/终点
    ax.plot(start_m[0], start_m[1], "o", color="#2ca02c", markersize=14,
            markeredgecolor="k", markeredgewidth=1.5, zorder=4, label="Start")
    ax.plot(goal_m[0], goal_m[1], "*", color="#d62728", markersize=18,
            markeredgecolor="k", markeredgewidth=1.0, zorder=4, label="Goal")

    # 比例尺
    fontprops = fm.FontProperties(family="serif", size=9)
    scalebar = AnchoredSizeBar(ax.transData, 5.0, "5 m", loc="lower right",
                               pad=0.4, borderpad=0.5, sep=4, frameon=True,
                               fontproperties=fontprops, size_vertical=0.12)
    ax.add_artist(scalebar)

    # 图例
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95, edgecolor="gray",
              fancybox=False, ncol=2, columnspacing=1.0, handletextpad=0.4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#aaaaaa")

    # 标题
    ax.set_title(
        f"Architecture Ablation — Run {run_idx}  "
        f"(goal_tol=0.3m, start=[{start_m[0]:.1f}, {start_m[1]:.1f}], "
        f"goal=[{goal_m[0]:.1f}, {goal_m[1]:.1f}])",
        fontsize=11, pad=8,
    )

    fig.tight_layout(pad=0.3)
    out_png = PROJ / "runs" / "pathviz" / f"ablation_arch_run{run_idx}_0.3m.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)
