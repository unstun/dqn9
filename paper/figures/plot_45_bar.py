"""4.5 AM x DQfD 组件消融 -- 指标柱状图 (2x2): SR / Path Length / Curvature / Compute Time。

数据源: runs202642/infer/abl_amdqfd_*/*/table2_kpis_mean.csv  (SR)
         runs202642/infer/abl_amdqfd_*/*/table2_kpis.csv       (Quality 过滤)

输出: paper/figures/fig_45_bar.{pdf,png}
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import apply_style, save_fig, RUNS2, C_FULL, C_NOAM, C_NODQFD

apply_style()

# ── 变体配置 ─────────────────────────────────────────────────────────
VARIANT_ORDER = ["Full (AM+DQfD)", "w/o DQfD", "w/o AM"]
DISPLAY = {
    "Full (AM+DQfD)": "Full\n(AM+DQfD)",
    "w/o DQfD":       "w/o\nDQfD",
    "w/o AM":         "w/o\nAM",
}
COLORS = {
    "Full (AM+DQfD)": C_FULL,
    "w/o DQfD":       C_NODQFD,
    "w/o AM":         C_NOAM,
}

# 目录名 -> 变体名
DIR_TO_VARIANT = {
    "abl_amdqfd_full":   "Full (AM+DQfD)",
    "abl_amdqfd_noAM":  "w/o AM",
    "abl_amdqfd_noDQfD": "w/o DQfD",
}

INFER_DIR = RUNS2 / "infer"

# ── 指标定义 ──────────────────────────────────────────────────────────
SUBPLOT_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def draw_single_bars(ax, vals, ylabel, fmt, subtitle, variant_order):
    """在 ax 上画单组柱状图。"""
    n = len(variant_order)
    x = np.arange(n)
    width = 0.55
    colors = [COLORS[v] for v in variant_order]

    bars = ax.bar(x, vals, width, color=colors, edgecolor="white",
                  linewidth=0.6, zorder=3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[v] for v in variant_order], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(subtitle, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_ylim(0, vals.max() * 1.22)


def main():
    # ── 读取 SR ──
    sr_data = {}
    for exp_dir in sorted(INFER_DIR.iterdir()):
        name = exp_dir.name
        if name not in DIR_TO_VARIANT:
            continue
        variant = DIR_TO_VARIANT[name]
        csv_files = sorted(exp_dir.glob("*/table2_kpis_mean.csv"))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[-1])
        sr_data[variant] = df["Success rate"].iloc[0] * 100

    # ── 读取 per-run 数据 ──
    all_variant_runs = {}
    for exp_dir in sorted(INFER_DIR.iterdir()):
        name = exp_dir.name
        if name not in DIR_TO_VARIANT:
            continue
        variant = DIR_TO_VARIANT[name]
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
        all_variant_runs[variant] = run_data

    # ── 3-variant filter ──
    all_runs = set()
    for d in all_variant_runs.values():
        all_runs.update(d.keys())
    filtered = [
        rid for rid in sorted(all_runs)
        if all(rid in all_variant_runs[v] and all_variant_runs[v][rid]["sr"] == 1.0
               for v in VARIANT_ORDER)
    ]
    N = len(filtered)
    print(f"Quality filter: N={N}")

    quality = {}
    for v in VARIANT_ORDER:
        d = all_variant_runs[v]
        quality[v] = {
            "pl": np.mean([d[i]["pl"] for i in filtered]),
            "curv": np.mean([d[i]["curv"] for i in filtered]),
            "time": np.mean([d[i]["time"] for i in filtered]),
        }

    # ── 组装指标数据 ──
    metrics = [
        ("Success Rate (%)", "{:.0f}",
         np.array([sr_data[v] for v in VARIANT_ORDER])),
        ("Path Length (m)", "{:.2f}",
         np.array([quality[v]["pl"] for v in VARIANT_ORDER])),
        ("Curvature (1/m)", "{:.4f}",
         np.array([quality[v]["curv"] for v in VARIANT_ORDER])),
        ("Compute Time (s)", "{:.3f}",
         np.array([quality[v]["time"] for v in VARIANT_ORDER])),
    ]

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.ravel()

    for i, (ylabel, fmt, vals) in enumerate(metrics):
        subtitle = f"{SUBPLOT_LABELS[i]} {ylabel}"
        draw_single_bars(axes[i], vals, ylabel, fmt, subtitle, VARIANT_ORDER)

    fig.tight_layout()
    save_fig(fig, "fig_45_bar")
    plt.close(fig)


if __name__ == "__main__":
    main()
