#!/usr/bin/env python3
"""生成 Full (AM+DQfD) vs w/o DQfD 训练曲线对比图。

子图 (a): 评估成功率 vs 回合数
子图 (b): 评估路径长度 vs 回合数（仅成功回合）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 数据路径 ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

FULL_EVAL = (
    ROOT / "runs" / "abl_diag10k_kt02_cnn_ddqn_md"
    / "train_20260315_120749" / "training_eval.csv"
)
NODQFD_EVAL = (
    ROOT / "runs" / "abl_amdqfd_noDQfD"
    / "train_20260316_001347" / "training_eval.csv"
)

# ── 读取数据 ─────────────────────────────────────────────────────────
df_full = pd.read_csv(FULL_EVAL)
df_nodqfd = pd.read_csv(NODQFD_EVAL)

# ── 绘图风格 ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

COLOR_FULL = "#2196F3"    # 蓝
COLOR_NODQFD = "#F44336"  # 红

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

# ── 滑动平均 ─────────────────────────────────────────────────────────
WINDOW = 10  # 10 个评估点 = 1000 回合窗口

def rolling_mean(series, window=WINDOW):
    return series.rolling(window=window, min_periods=1).mean()

# ── (a) 成功率 vs 回合数 ─────────────────────────────────────────────
ax = axes[0]
# 原始数据（淡色散点）
ax.scatter(df_full["episode"], df_full["success_rate"],
           color=COLOR_FULL, s=8, alpha=0.2, zorder=2)
ax.scatter(df_nodqfd["episode"], df_nodqfd["success_rate"],
           color=COLOR_NODQFD, s=8, alpha=0.2, zorder=2)
# 滑动平均（实线）
ax.plot(df_full["episode"], rolling_mean(df_full["success_rate"]),
        "-", color=COLOR_FULL, linewidth=1.8, label="Full (AM + DQfD)", zorder=3)
ax.plot(df_nodqfd["episode"], rolling_mean(df_nodqfd["success_rate"]),
        "-", color=COLOR_NODQFD, linewidth=1.8, label="w/o DQfD", zorder=3)
ax.set_xlabel("Training episode")
ax.set_ylabel("Evaluation success rate")
ax.set_ylim(-0.05, 1.1)
ax.set_xlim(0, 10200)
ax.legend(loc="center right", framealpha=0.9)
ax.set_title("(a) Success rate", fontsize=10)
ax.grid(True, alpha=0.3, linewidth=0.5)

# ── (b) 路径长度 vs 回合数 ───────────────────────────────────────────
ax = axes[1]

# 仅绘制成功回合（有路径长度的数据点）
full_ok = df_full[df_full["success_rate"] > 0].copy()
nodqfd_ok = df_nodqfd[df_nodqfd["success_rate"] > 0].copy()

# 原始数据（淡色散点）
ax.scatter(full_ok["episode"], full_ok["avg_path_length"],
           color=COLOR_FULL, s=8, alpha=0.2, zorder=2)
ax.scatter(nodqfd_ok["episode"], nodqfd_ok["avg_path_length"],
           color=COLOR_NODQFD, s=8, alpha=0.2, zorder=2)
# 滑动平均（实线）
ax.plot(full_ok["episode"], rolling_mean(full_ok["avg_path_length"]),
        "-", color=COLOR_FULL, linewidth=1.8, label="Full (AM + DQfD)", zorder=3)
ax.plot(nodqfd_ok["episode"], rolling_mean(nodqfd_ok["avg_path_length"]),
        "-", color=COLOR_NODQFD, linewidth=1.8, label="w/o DQfD", zorder=3)
ax.set_xlabel("Training episode")
ax.set_ylabel("Evaluation path length (m)")
ax.set_xlim(0, 10200)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_title("(b) Path length", fontsize=10)
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()

# ── 保存 ─────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent
fig.savefig(str(OUT / "fig_training_curve.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.1)
fig.savefig(str(OUT / "fig_training_curve.pdf"), dpi=300,
            bbox_inches="tight", pad_inches=0.1)
print(f"Saved: {OUT / 'fig_training_curve.pdf'}")
plt.close(fig)
