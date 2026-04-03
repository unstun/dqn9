"""Chapter 4 统一绘图样式与工具函数。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 项目路径 ──────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent
RUNS = PROJ / "runs"
RUNS2 = PROJ / "runs202642"
RUNS3 = PROJ / "runs202643"
RESULTS = PROJ / "paper" / "results"

# ── 统一 rcParams ────────────────────────────────────────────────────
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """应用统一 rcParams。"""
    plt.rcParams.update(RCPARAMS)


# ── 配色 ─────────────────────────────────────────────────────────────
# 4.3 核心对比
C_MDDDQN = "#1f77b4"
C_HASTAR = "#ff7f0e"
C_RRTSTAR = "#2ca02c"

# 4.5 组件消融
C_FULL = "#2196F3"
C_NOAM = "#FF9800"
C_NODQFD = "#F44336"

# 4.4 架构消融 (4 DQN 变体, MinTD)
ARCH_COLORS = {
    "MD-DQN":    "#1f77b4",
    "Duel-DQN":  "#ff7f0e",
    "MHA-DQN":   "#2ca02c",
    "DQN":       "#d62728",
}

# 训练目录后缀 → 显示名
ARCH_DIR_TO_LABEL = {
    "cnn_dqn_md":    "MD-DQN",
    "cnn_dqn_duel":  "Duel-DQN",
    "cnn_dqn_mha":   "MHA-DQN",
    "cnn_dqn":       "DQN",
}

# CSV Variant 名 → 显示名
ARCH_CSV_TO_LABEL = {
    "CNN-DQN+MD":    "MD-DQN",
    "CNN-DQN+Duel":  "Duel-DQN",
    "CNN-DQN+MHA":   "MHA-DQN",
    "CNN-DQN":       "DQN",
}


# ── 通用工具 ─────────────────────────────────────────────────────────
def rolling_mean(series: pd.Series, window: int = 50) -> pd.Series:
    """滑动均值平滑。"""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def save_fig(fig: plt.Figure, name: str):
    """同时保存 PDF 和 PNG 到 paper/figures/。"""
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{name}.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {FIG_DIR / name}.pdf")
