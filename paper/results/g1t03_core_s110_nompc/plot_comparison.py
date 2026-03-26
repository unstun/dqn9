"""g1t03 核心对比：per-run 质量曲线 (纯规划, 无 MPC)"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

ALGO_MAP = {
    "CNN-DDQN+Duel": "MD-DDQN (Ours)",
    "Hybrid A* (Dang 2022)": "Hybrid A*",
    "RRT* (Yoon 2018)": "RRT*",
}
ALGO_ORDER = ["MD-DDQN (Ours)", "Hybrid A*", "RRT*"]
COLORS = {"MD-DDQN (Ours)": "#1f77b4", "Hybrid A*": "#ff7f0e", "RRT*": "#2ca02c"}
MARKERS = {"MD-DDQN (Ours)": "o", "Hybrid A*": "s", "RRT*": "^"}

METRICS = [
    ("avg_path_length", "Path Length (m)", False),
    ("avg_curvature_1_m", "Curvature (1/m)", False),
    ("planning_time_s", "Compute Time (s)", True),  # log scale
]


def load_and_filter(suite: str):
    """加载 CSV，筛选三算法共同成功的 run"""
    df = pd.read_csv(HERE / suite / "table2_kpis_raw.csv")
    df["algo"] = df["Algorithm"].map(ALGO_MAP)

    # 找共同成功的 run_idx
    pivot = df.pivot_table(index="run_idx", columns="algo",
                           values="success_rate", aggfunc="first")
    common = pivot.dropna().index[pivot.dropna().min(axis=1) == 1.0]
    df_ok = df[df["run_idx"].isin(common) & (df["success_rate"] == 1.0)].copy()
    df_ok = df_ok.sort_values("run_idx")
    return df_ok, len(common)


def plot_suite(ax_row, df, n_common, suite_label):
    """在一行 axes 上画 3 个指标的 per-run 对比线"""
    for col_idx, (metric, ylabel, use_log) in enumerate(METRICS):
        ax = ax_row[col_idx]
        for algo in ALGO_ORDER:
            sub = df[df["algo"] == algo].sort_values("run_idx")
            xs = np.arange(len(sub))
            ax.plot(xs, sub[metric].values,
                    color=COLORS[algo], marker=MARKERS[algo],
                    markersize=4, linewidth=1.2, label=algo, alpha=0.85)
        ax.set_xlabel("Run index (common-success)")
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")
        ax.set_title(f"{suite_label} — {ylabel}")
        if col_idx == 0:
            ax.legend(loc="best", framealpha=0.8)
        ax.grid(True, alpha=0.3)


# ── 主图：2 行 × 3 列 ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("g1t03 Core Comparison — Per-Run Quality (Pure Planning, No MPC)",
             fontsize=14, fontweight="bold")

for row_idx, (suite, label) in enumerate([
    ("long", "Long Distance (≥18m)"),
    ("short", "Short Distance (6–14m)"),
]):
    df_ok, n = load_and_filter(suite)
    plot_suite(axes[row_idx], df_ok, n, f"{label}  [n={n}]")

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = HERE / "g1t03_core_comparison_curves.png"
fig.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")

# ── 额外：SR 柱状图 ──
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
fig2.suptitle("g1t03 Core — Success Rate (50 runs, goal_tol=0.3m)", fontsize=13, fontweight="bold")

for i, (suite, label) in enumerate([("long", "Long (≥18m)"), ("short", "Short (6–14m)")]):
    df = pd.read_csv(HERE / suite / "table2_kpis_raw.csv")
    df["algo"] = df["Algorithm"].map(ALGO_MAP)
    sr = df.groupby("algo")["success_rate"].mean().reindex(ALGO_ORDER) * 100
    bars = axes2[i].bar(ALGO_ORDER, sr, color=[COLORS[a] for a in ALGO_ORDER], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, sr):
        axes2[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                      f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
    axes2[i].set_ylim(0, 105)
    axes2[i].set_ylabel("Success Rate (%)")
    axes2[i].set_title(label)
    axes2[i].grid(axis="y", alpha=0.3)

fig2.tight_layout(rect=[0, 0, 1, 0.93])
out2 = HERE / "g1t03_core_sr_bar.png"
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved → {out2}")

plt.show()
