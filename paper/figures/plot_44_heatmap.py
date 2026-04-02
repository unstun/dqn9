"""4.4 节 热力图: 8 种 CNN-DRL 架构变体 x 4 项指标归一化对比。

行按平均归一化得分降序排列, MD-DDQN 行外加蓝色边框高亮。

数据源: runs202642/infer/abl_arch_*/*/table2_kpis_mean.csv  (SR)
         runs202642/infer/abl_arch_*/*/table2_kpis.csv       (Quality 过滤)
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from style import apply_style, save_fig, RUNS2, ARCH_DIR_TO_LABEL

apply_style()

# ── 数据读取 ─────────────────────────────────────────────────────────
INFER_DIR = RUNS2 / "infer"

# 目录名前缀 -> 显示名
DIR_PREFIX_TO_LABEL = {f"abl_arch_{k}": v for k, v in ARCH_DIR_TO_LABEL.items()}

# 读取 SR (from mean CSV)
sr_data = {}
for exp_dir in sorted(INFER_DIR.iterdir()):
    name = exp_dir.name
    if name not in DIR_PREFIX_TO_LABEL:
        continue
    label = DIR_PREFIX_TO_LABEL[name]
    csv_files = sorted(exp_dir.glob("*/table2_kpis_mean.csv"))
    if not csv_files:
        continue
    df = pd.read_csv(csv_files[-1])
    sr_data[label] = df["Success rate"].iloc[0] * 100  # %

# 读取 per-run 数据 (for quality filter)
all_variant_runs = {}
for exp_dir in sorted(INFER_DIR.iterdir()):
    name = exp_dir.name
    if name not in DIR_PREFIX_TO_LABEL:
        continue
    label = DIR_PREFIX_TO_LABEL[name]
    csv_files = sorted(exp_dir.glob("*/table2_kpis.csv"))
    if not csv_files:
        continue
    df = pd.read_csv(csv_files[-1])
    run_data = {}
    for _, row in df.iterrows():
        rid = int(row["Run index"])
        sr = float(row["Success rate"])
        run_data[rid] = {
            "sr": sr,
            "pl": float(row["Average path length (m)"]) if sr == 1.0 else None,
            "curv": float(row["Average curvature (1/m)"]) if sr == 1.0 else None,
            "time": float(row["Compute time (s)"]) if sr == 1.0 else None,
        }
    all_variant_runs[label] = run_data

# 8-variant filter for quality
variant_labels = sorted(all_variant_runs.keys())
all_runs = set()
for d in all_variant_runs.values():
    all_runs.update(d.keys())

filtered_runs = [
    rid for rid in sorted(all_runs)
    if all(rid in all_variant_runs[v] and all_variant_runs[v][rid]["sr"] == 1.0
           for v in variant_labels)
]
N = len(filtered_runs)
print(f"Quality filter: N={N}")

quality_data = {}
for v in variant_labels:
    d = all_variant_runs[v]
    pls = [d[i]["pl"] for i in filtered_runs]
    curvs = [d[i]["curv"] for i in filtered_runs]
    times = [d[i]["time"] for i in filtered_runs]
    quality_data[v] = {
        "pl": sum(pls) / len(pls),
        "curv": sum(curvs) / len(curvs),
        "time": sum(times) / len(times),
    }

# ── 组装矩阵 (8 行 x 4 列) ──────────────────────────────────────────
COL_NAMES = ["SR", "PL", "Curv", "Time"]
COL_DISPLAY = [f"SR (%)", f"PL (m)\n(N={N})", f"Curv (1/m)\n(N={N})", f"Time (s)\n(N={N})"]

raw_vals = np.zeros((len(variant_labels), 4))
for i, v in enumerate(variant_labels):
    raw_vals[i, 0] = sr_data[v]
    raw_vals[i, 1] = quality_data[v]["pl"]
    raw_vals[i, 2] = quality_data[v]["curv"]
    raw_vals[i, 3] = quality_data[v]["time"]

# ── 归一化: [0, 1], 1 = best ──────────────────────────────────────────
HIGHER_BETTER = {0}  # SR: higher is better; PL, Curv, Time: lower is better

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
labels_sorted = np.array(variant_labels)[sort_idx]

# ── 绘图 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

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

        if col == "SR":
            txt = f"{int(rv)}%"
        elif col == "PL":
            txt = f"{rv:.3f}"
        elif col == "Curv":
            txt = f"{rv:.4f}"
        else:
            txt = f"{rv:.3f}"

        tc = "white" if nv < 0.3 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=tc)

# ── MD-DDQN 行蓝色高亮框 ──────────────────────────────────────────────
md_row = int(np.where(labels_sorted == "MD-DDQN")[0][0])
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
