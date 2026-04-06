"""4.3 核心对比 -- 指标柱状图 (2x2): SR / Path Length / Curvature / Compute Time.

数据源: runs202643/infer/core_baseline_dqn_sr_{long,short}/*/table2_kpis_mean.csv
输出:   paper/figures/fig_43_bar.{pdf,png}
"""

from __future__ import annotations

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import apply_style, save_fig, RUNS3, C_MDDDQN, C_HASTAR, C_RRTSTAR

apply_style()

# ── 算法映射 ─────────────────────────────────────────────────────────
ALGO_ORDER = ["CNN-DQN+Duel", "Hybrid A*", "RRT*"]
DISPLAY = {
    "CNN-DQN+Duel": "MD-DQN\n(Ours)",
    "Hybrid A*":    "Improved\nHA*",
    "RRT*":         "Spline-\nRRT*",
}
COLORS = {
    "CNN-DQN+Duel": C_MDDDQN,
    "Hybrid A*":    C_HASTAR,
    "RRT*":         C_RRTSTAR,
}


def load_csv(dist: str) -> pd.DataFrame:
    """读取 table2_kpis_mean.csv 并按 ALGO_ORDER 排序。"""
    pattern = str(RUNS3 / "infer" / f"core_baseline_dqn_sr_{dist}" / "*" / "table2_kpis_mean.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV found for {dist}: {pattern}")
    df = pd.read_csv(files[-1])
    df = df.set_index("Algorithm name").loc[ALGO_ORDER].reset_index()
    return df


# ── 指标定义 ──────────────────────────────────────────────────────────
# (csv_col, display_label, multiplier, fmt)
METRICS = [
    ("Success rate",             "Success Rate (%)",     100,  "{:.0f}"),
    ("Average path length (m)",  "Path Length (m)",        1,  "{:.1f}"),
    ("Average curvature (1/m)",  "Curvature (1/m)",        1,  "{:.3f}"),
    ("Compute time (s)",         "Compute Time (s)",       1,  "{:.2f}"),
]
SUBPLOT_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def draw_grouped_bars(ax: plt.Axes, vals_long: np.ndarray,
                      vals_short: np.ndarray, ylabel: str,
                      fmt: str, subtitle: str) -> None:
    """在 ax 上画 3 组 x2 (Long/Short) 分组柱状图。"""
    n = len(ALGO_ORDER)
    x = np.arange(n)
    width = 0.32

    colors = [COLORS[a] for a in ALGO_ORDER]

    # Long -- 实心
    bars_l = ax.bar(x - width / 2, vals_long, width,
                    color=colors, edgecolor="white", linewidth=0.6,
                    label="Long", zorder=3)
    # Short -- 斜线填充
    bars_s = ax.bar(x + width / 2, vals_short, width,
                    color=colors, edgecolor="white", linewidth=0.6,
                    hatch="///", alpha=0.75, label="Short", zorder=3)

    # 数值标注
    for bars in (bars_l, bars_s):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    fmt.format(h), ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[a] for a in ALGO_ORDER], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(subtitle, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    # y 轴留出标注空间
    ymax = max(vals_long.max(), vals_short.max())
    ax.set_ylim(0, ymax * 1.22)


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    df_long  = load_csv("long")
    df_short = load_csv("short")

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.ravel()

    for i, (col, ylabel, mult, fmt) in enumerate(METRICS):
        vals_l = df_long[col].values * mult
        vals_s = df_short[col].values * mult
        subtitle = f"{SUBPLOT_LABELS[i]} {ylabel}"
        draw_grouped_bars(axes[i], vals_l, vals_s, ylabel, fmt, subtitle)

    # 统一图例 -- Long(实心) / Short(斜线)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", edgecolor="white", label="Long"),
        Patch(facecolor="gray", edgecolor="white", hatch="///",
              alpha=0.75, label="Short"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=8)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "fig_43_bar")
    plt.close(fig)


if __name__ == "__main__":
    main()
