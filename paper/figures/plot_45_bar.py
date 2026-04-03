"""4.5 AM x DQfD 组件消融 -- 指标柱状图 (2x2): SR / Path Length / Curvature / Compute Time。

每个子图显示 Short 和 Long 距离的对比（分组柱状图）。

数据源: runs202643/infer/  (MinTD, MD-DQN 为基底)

输出: paper/figures/fig_45_bar.{pdf,png}
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import apply_style, save_fig, RUNS3, C_FULL, C_NOAM, C_NODQFD

apply_style()

# ── 变体配置 ─────────────────────────────────────────────────────────
VARIANT_ORDER = ["Full (AM+DQfD)", "w/o AM", "w/o DQfD"]
DISPLAY = {
    "Full (AM+DQfD)": "Full\n(AM+DQfD)",
    "w/o AM":         "w/o\nAM",
    "w/o DQfD":       "w/o\nDQfD",
}
COLORS = {
    "Full (AM+DQfD)": C_FULL,
    "w/o AM":         C_NOAM,
    "w/o DQfD":       C_NODQFD,
}

# 目录名 -> 变体名
SHORT_DIRS = {
    "abl_minloss_cnn_dqn_md":        "Full (AM+DQfD)",
    "abl_minloss_amdqfd_dqn_noAM":   "w/o AM",
    "abl_amdqfd_dqn_infer_noDQfD":   "w/o DQfD",
}
LONG_DIRS = {
    "abl_minloss_cnn_dqn_md_long":        "Full (AM+DQfD)",
    "abl_minloss_amdqfd_dqn_noAM_long":   "w/o AM",
    "abl_amdqfd_dqn_infer_noDQfD_long":   "w/o DQfD",
}

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


def quality_filter(dir_map: dict, variants: list) -> tuple[int, dict]:
    all_data = {}
    for dname, vname in dir_map.items():
        all_data[vname] = read_runs(dname)

    all_rids = set()
    for d in all_data.values():
        all_rids.update(d.keys())

    filtered = [
        rid for rid in sorted(all_rids)
        if all(rid in all_data[v] and all_data[v][rid]["sr"] == 1.0
               for v in variants)
    ]
    N = len(filtered)
    quality = {}
    for v in variants:
        d = all_data[v]
        quality[v] = {
            "pl": np.mean([d[i]["pl"] for i in filtered]),
            "curv": np.mean([d[i]["curv"] for i in filtered]),
            "time": np.mean([d[i]["time"] for i in filtered]),
        }
    return N, quality


def draw_grouped_bars(ax, short_vals, long_vals, ylabel, fmt, subtitle,
                      variant_order, n_short=None, n_long=None):
    n = len(variant_order)
    x = np.arange(n)
    width = 0.30
    colors = [COLORS[v] for v in variant_order]

    # Long 实色, Short 斜线
    bars_long = ax.bar(x - width / 2, long_vals, width,
                       color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    bars_short = ax.bar(x + width / 2, short_vals, width,
                        color=colors, edgecolor="grey", linewidth=0.6,
                        alpha=0.55, hatch="//", zorder=3)

    for bar in bars_long:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va="bottom", fontsize=6.5)
    for bar in bars_short:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[v] for v in variant_order], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(subtitle, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    all_vals = np.concatenate([short_vals, long_vals])
    ax.set_ylim(0, all_vals.max() * 1.25)


def main():
    # ── SR ──
    sr_short = {SHORT_DIRS[d]: read_sr(d) for d in SHORT_DIRS}
    sr_long = {LONG_DIRS[d]: read_sr(d) for d in LONG_DIRS}

    # ── Quality ──
    N_short, q_short = quality_filter(SHORT_DIRS, VARIANT_ORDER)
    N_long, q_long = quality_filter(LONG_DIRS, VARIANT_ORDER)
    print(f"Quality Short N={N_short}, Long N={N_long}")

    # ── 组装指标数据 ──
    metrics = [
        ("Success Rate (%)", "{:.0f}",
         np.array([sr_short[v] for v in VARIANT_ORDER]),
         np.array([sr_long[v] for v in VARIANT_ORDER])),
        (f"Path Length (m)", "{:.2f}",
         np.array([q_short[v]["pl"] for v in VARIANT_ORDER]),
         np.array([q_long[v]["pl"] for v in VARIANT_ORDER])),
        (f"Curvature (1/m)", "{:.4f}",
         np.array([q_short[v]["curv"] for v in VARIANT_ORDER]),
         np.array([q_long[v]["curv"] for v in VARIANT_ORDER])),
        (f"Compute Time (s)", "{:.3f}",
         np.array([q_short[v]["time"] for v in VARIANT_ORDER]),
         np.array([q_long[v]["time"] for v in VARIANT_ORDER])),
    ]

    SUBPLOT_LABELS = ["(a)", "(b)", "(c)", "(d)"]

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.ravel()

    for i, (ylabel, fmt, s_vals, l_vals) in enumerate(metrics):
        subtitle = f"{SUBPLOT_LABELS[i]} {ylabel}"
        draw_grouped_bars(axes[i], s_vals, l_vals, ylabel, fmt, subtitle,
                          VARIANT_ORDER, N_short, N_long)

    # ── 图例 ──
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="grey", alpha=0.8, label="Long ($\\geq$18 m)"),
        Patch(facecolor="grey", alpha=0.55, hatch="//",
              edgecolor="grey", label="Short (6\u201314 m)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "fig_45_bar")
    plt.close(fig)


if __name__ == "__main__":
    main()
