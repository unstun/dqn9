"""4.4 节 热力图: 4 种 DQN 架构变体 x 5 项指标归一化对比。

行按平均归一化得分降序排列, MD-DQN 行外加蓝色边框高亮。

数据源: runs202643/infer/abl_minloss_cnn_*  (Short SR + Quality)
         runs202643/infer/abl_minloss_cnn_*_long  (Long SR + Quality)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from style import apply_style, save_fig, RUNS3

apply_style()

# ── 变体与目录映射 ──────────────────────────────────────────────────────
VARIANTS = ["MD-DQN", "Duel-DQN", "MHA-DQN", "DQN"]

SHORT_DIRS = {
    "MD-DQN":   "abl_minloss_cnn_dqn_md",
    "Duel-DQN": "abl_minloss_cnn_dqn_duel",
    "MHA-DQN":  "abl_minloss_cnn_dqn_mha",
    "DQN":      "abl_minloss_cnn_dqn",
}
LONG_DIRS = {k: v + "_long" for k, v in SHORT_DIRS.items()}

INFER_DIR = RUNS3 / "infer"


def read_sr(dir_name: str) -> float:
    d = INFER_DIR / dir_name
    csvs = sorted(d.glob("*/table2_kpis_mean.csv"))
    if not csvs:
        raise FileNotFoundError(f"No mean CSV in {d}")
    df = pd.read_csv(csvs[-1])
    return df["Success rate"].iloc[0] * 100


def read_runs(dir_name: str) -> dict:
    d = INFER_DIR / dir_name
    csvs = sorted(d.glob("*/table2_kpis.csv"))
    if not csvs:
        return {}
    df = pd.read_csv(csvs[-1])
    runs = {}
    for _, r in df.iterrows():
        rid = int(r["Run index"])
        sr = float(r["Success rate"])
        runs[rid] = {
            "sr": sr,
            "pl": float(r["Average path length (m)"]) if sr == 1.0 else None,
            "curv": float(r["Average curvature (1/m)"]) if sr == 1.0 else None,
            "time": float(r["Compute time (s)"]) if sr == 1.0 else None,
        }
    return runs


# ── 读取 SR ──────────────────────────────────────────────────────────
sr_short = {v: read_sr(SHORT_DIRS[v]) for v in VARIANTS}
sr_long = {v: read_sr(LONG_DIRS[v]) for v in VARIANTS}

# ── 读取 per-run 并计算质量子集 (Long, N=?) ──────────────────────────
long_runs = {v: read_runs(LONG_DIRS[v]) for v in VARIANTS}

all_rids = set()
for d in long_runs.values():
    all_rids.update(d.keys())

filtered = [
    rid for rid in sorted(all_rids)
    if all(rid in long_runs[v] and long_runs[v][rid]["sr"] == 1.0
           for v in VARIANTS)
]
N = len(filtered)
print(f"Quality filter (Long): N={N}")

quality = {}
for v in VARIANTS:
    d = long_runs[v]
    quality[v] = {
        "pl": np.mean([d[i]["pl"] for i in filtered]),
        "curv": np.mean([d[i]["curv"] for i in filtered]),
        "time": np.mean([d[i]["time"] for i in filtered]),
    }

# ── 组装矩阵 (4 行 x 5 列) ──────────────────────────────────────────
COL_NAMES = ["SR_S", "SR_L", "PL", "Curv", "Time"]
COL_DISPLAY = [
    "Short\nSR (%)",
    "Long\nSR (%)",
    f"PL (m)\n(N={N})",
    f"Curv\n(N={N})",
    f"Time (s)\n(N={N})",
]

raw_vals = np.zeros((len(VARIANTS), 5))
for i, v in enumerate(VARIANTS):
    raw_vals[i, 0] = sr_short[v]
    raw_vals[i, 1] = sr_long[v]
    raw_vals[i, 2] = quality[v]["pl"]
    raw_vals[i, 3] = quality[v]["curv"]
    raw_vals[i, 4] = quality[v]["time"]

# ── 归一化: [0, 1], 1 = best ──────────────────────────────────────────
HIGHER_BETTER = {0, 1}  # SR: higher is better

norm_vals = np.zeros_like(raw_vals)
for j in range(raw_vals.shape[1]):
    v = raw_vals[:, j]
    vmin, vmax = v.min(), v.max()
    span = vmax - vmin if vmax != vmin else 1.0
    if j in HIGHER_BETTER:
        norm_vals[:, j] = (v - vmin) / span
    else:
        norm_vals[:, j] = (vmax - v) / span

# ── 按平均归一化得分降序排列 ───────────────────────────────────────────
avg_scores = norm_vals.mean(axis=1)
sort_idx = np.argsort(-avg_scores)

raw_vals = raw_vals[sort_idx]
norm_vals = norm_vals[sort_idx]
labels_sorted = np.array(VARIANTS)[sort_idx]

# ── 绘图 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3))

im = ax.imshow(norm_vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(np.arange(len(COL_DISPLAY)))
ax.set_xticklabels(COL_DISPLAY, fontsize=9)
ax.set_yticks(np.arange(len(labels_sorted)))
ax.set_yticklabels(labels_sorted, fontsize=9)

ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

# ── 单元格标注 (原始值) ────────────────────────────────────────────────
for i in range(norm_vals.shape[0]):
    for j in range(norm_vals.shape[1]):
        nv = norm_vals[i, j]
        rv = raw_vals[i, j]
        col = COL_NAMES[j]

        if col in ("SR_S", "SR_L"):
            txt = f"{int(rv)}%"
        elif col == "PL":
            txt = f"{rv:.3f}"
        elif col == "Curv":
            txt = f"{rv:.4f}"
        else:
            txt = f"{rv:.3f}"

        tc = "white" if nv < 0.3 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=tc)

# ── MD-DQN 行蓝色高亮框 ──────────────────────────────────────────────
md_row = int(np.where(labels_sorted == "MD-DQN")[0][0])
rect = mpatches.FancyBboxPatch(
    (-0.5, md_row - 0.5),
    len(COL_NAMES),
    1,
    boxstyle="square,pad=0",
    edgecolor="#2166ac",
    facecolor="none",
    linewidth=3,
    clip_on=False,
)
ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Normalized Score (1 = best)", fontsize=9)

fig.tight_layout()
save_fig(fig, "fig_44_heatmap")
plt.close(fig)
