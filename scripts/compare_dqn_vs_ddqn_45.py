#!/usr/bin/env python3
# ============================================================================
# compare_dqn_vs_ddqn_45.py
# ----------------------------------------------------------------------------
# §4.5 full-factor ablation: cnn-dqn (vanilla DQN) vs cnn-ddqn (Double DQN).
#
# Reads:
#   <dqn-runs-root>/<out>/<latest_TS>/table2_kpis_mean.csv          (24 cells)
#   <ddqn-runs-root>/infer/<out>/<latest_TS>/table2_kpis_mean.csv   (24 cells)
#                              ^^^^^^^ DDQN layout has an extra "infer/" layer
#
# Output (under --out-dir):
#   ablation_20260408_dqn_vs_ddqn_base.md   markdown report
#   raw_48.csv                              48-row tidy table for re-analysis
#   figs/dqn_vs_ddqn_sr_short.png           grouped bar SR (4 var x 3 cond x 2 base)
#   figs/dqn_vs_ddqn_sr_long.png            same, long distance
#   figs/dqn_vs_ddqn_pathlen.png            path length both distances
# ============================================================================

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Factor enumeration
# ----------------------------------------------------------------------------
VARIANTS = [
    ("",            "vanilla"),
    ("_duel",       "duel"),
    ("_munch",      "munch"),
    ("_munch_duel", "munch_duel"),
]
CONDITIONS = ["full", "noAM", "noDQfD"]
DISTANCES = ["short", "long"]


def enumerate_cells():
    for v_suf, v_label in VARIANTS:
        for cond in CONDITIONS:
            for dist in DISTANCES:
                yield {
                    "variant":   v_label,
                    "condition": cond,
                    "distance":  dist,
                    "v_suf":     v_suf,
                }


def latest_ts_subdir(p: Path) -> Path | None:
    if not p.exists():
        return None
    cands = [c for c in p.iterdir() if c.is_dir() and c.name[:8].isdigit()]
    if not cands:
        return None
    return max(cands, key=lambda c: c.name)


def load_one(runs_root: Path, base: str, cell: dict, nested_infer: bool):
    out_name = (
        f"abl_amdqfd_{base}{cell['v_suf']}_infer_"
        f"{cell['condition']}_sr_{cell['distance']}"
    )
    cell_dir = (runs_root / "infer" / out_name) if nested_infer else (runs_root / out_name)
    ts_dir = latest_ts_subdir(cell_dir)
    if ts_dir is None:
        return None
    csv = ts_dir / "table2_kpis_mean.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "variant":   cell["variant"],
        "condition": cell["condition"],
        "distance":  cell["distance"],
        "base":      "DQN" if base == "cnn_dqn" else "DDQN",
        "ts":        ts_dir.name,
        "algo":      row.get("Algorithm name", ""),
        "sr":        float(row.get("Success rate", float("nan"))),
        "path_len":  float(row.get("Average path length (m)", float("nan"))),
        "path_time": float(row.get("Path time (s)", float("nan"))),
        "curv":      float(row.get("Average curvature (1/m)", float("nan"))),
        "plan_t":    float(row.get("Planning time (s)", float("nan"))),
        "comp_t":    float(row.get("Compute time (s)", float("nan"))),
    }


def gather(dqn_root: Path, ddqn_root: Path) -> pd.DataFrame:
    rows = []
    miss_dqn, miss_ddqn = [], []
    for cell in enumerate_cells():
        d = load_one(dqn_root, "cnn_dqn", cell, nested_infer=False)
        if d is not None:
            rows.append(d)
        else:
            miss_dqn.append(cell)
        e = load_one(ddqn_root, "cnn_ddqn", cell, nested_infer=True)
        if e is not None:
            rows.append(e)
        else:
            miss_ddqn.append(cell)
    if miss_dqn:
        print(f"[compare] missing DQN cells: {len(miss_dqn)}")
        for c in miss_dqn:
            print(f"  - {c['variant']} / {c['condition']} / {c['distance']}")
    if miss_ddqn:
        print(f"[compare] missing DDQN cells: {len(miss_ddqn)}")
        for c in miss_ddqn:
            print(f"  - {c['variant']} / {c['condition']} / {c['distance']}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
DQN_COLOR  = "#2c7fb8"
DDQN_COLOR = "#e34a33"


def _grouped_bar(ax, df, distance, value_col, ylabel, ylim=None):
    var_order = [v[1] for v in VARIANTS]
    sub = df[df["distance"] == distance]
    n = len(var_order) * len(CONDITIONS)
    x = np.arange(n)
    width = 0.38
    dqn_vals, ddqn_vals, labels = [], [], []
    for cond in CONDITIONS:
        for var in var_order:
            cell = sub[(sub["variant"] == var) & (sub["condition"] == cond)]
            d = cell[cell["base"] == "DQN"][value_col]
            e = cell[cell["base"] == "DDQN"][value_col]
            dqn_vals.append(float(d.iloc[0]) if not d.empty else np.nan)
            ddqn_vals.append(float(e.iloc[0]) if not e.empty else np.nan)
            labels.append(f"{var}\n{cond}")
    ax.bar(x - width / 2, dqn_vals,  width, label="cnn-dqn",  color=DQN_COLOR)
    ax.bar(x + width / 2, ddqn_vals, width, label="cnn-ddqn", color=DDQN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_sr(df: pd.DataFrame, distance: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _grouped_bar(ax, df, distance, "sr", "Success rate", ylim=(0, 1.05))
    ax.set_title(
        f"\u00a74.5 cnn-dqn vs cnn-ddqn  --  Success rate ({distance} distance)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pathlen(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, dist in zip(axes, DISTANCES):
        _grouped_bar(ax, df, dist, "path_len", "Mean path length (m)")
        ax.set_title(f"Path length -- {dist}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------------
def md_table_metric(df: pd.DataFrame, metric: str, distance: str, fmt: str = "{:.2f}"):
    var_order = [v[1] for v in VARIANTS]
    sub = df[df["distance"] == distance]
    header = ("| variant | full (DQN/DDQN, \u0394) | "
              "noAM (DQN/DDQN, \u0394) | noDQfD (DQN/DDQN, \u0394) |")
    sep = "|---|---|---|---|"
    lines = [header, sep]
    for var in var_order:
        cells = []
        for cond in CONDITIONS:
            row = sub[(sub["variant"] == var) & (sub["condition"] == cond)]
            d = row[row["base"] == "DQN"][metric]
            e = row[row["base"] == "DDQN"][metric]
            ds = fmt.format(float(d.iloc[0])) if not d.empty else "—"
            es = fmt.format(float(e.iloc[0])) if not e.empty else "—"
            if (not d.empty) and (not e.empty):
                delta = float(d.iloc[0]) - float(e.iloc[0])
                delta_s = f"{'+' if delta >= 0 else ''}{delta:+.2f}".replace("++", "+")
                cells.append(f"{ds} / {es} ({delta_s})")
            else:
                cells.append(f"{ds} / {es}")
        lines.append(f"| {var} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare cnn-dqn vs cnn-ddqn full-factor §4.5 ablation"
    )
    ap.add_argument("--dqn-runs-root",  required=True, type=Path)
    ap.add_argument("--ddqn-runs-root", required=True, type=Path)
    ap.add_argument("--out-dir",        required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] dqn_root  = {args.dqn_runs_root}")
    print(f"[compare] ddqn_root = {args.ddqn_runs_root}")
    print(f"[compare] out_dir   = {out_dir}")

    df = gather(args.dqn_runs_root.resolve(), args.ddqn_runs_root.resolve())
    n_dqn  = (df["base"] == "DQN").sum()
    n_ddqn = (df["base"] == "DDQN").sum()
    print(f"[compare] gathered {len(df)} rows  (DQN={n_dqn}, DDQN={n_ddqn})")

    df.to_csv(out_dir / "raw_48.csv", index=False)

    # ---- figures ----
    plot_sr(df, "short", fig_dir / "dqn_vs_ddqn_sr_short.png")
    plot_sr(df, "long",  fig_dir / "dqn_vs_ddqn_sr_long.png")
    plot_pathlen(df, fig_dir / "dqn_vs_ddqn_pathlen.png")

    # ---- pivot for auto findings ----
    pivot = df.pivot_table(
        index=["variant", "condition", "distance"],
        columns="base", values="sr",
    ).reset_index()
    if "DQN" in pivot.columns and "DDQN" in pivot.columns:
        pivot["delta_sr"] = pivot["DQN"] - pivot["DDQN"]
    else:
        pivot["delta_sr"] = float("nan")
    pivot_clean = pivot.dropna(subset=["delta_sr"])
    n_clean = len(pivot_clean)

    if n_clean > 0:
        avg_delta     = pivot_clean["delta_sr"].mean()
        n_dqn_better  = int((pivot_clean["delta_sr"] >  0.02).sum())
        n_ddqn_better = int((pivot_clean["delta_sr"] < -0.02).sum())
        n_tied        = n_clean - n_dqn_better - n_ddqn_better
        biggest_dqn   = pivot_clean.nlargest(3, "delta_sr")[
            ["variant", "condition", "distance", "DQN", "DDQN", "delta_sr"]
        ]
        biggest_ddqn  = pivot_clean.nsmallest(3, "delta_sr")[
            ["variant", "condition", "distance", "DQN", "DDQN", "delta_sr"]
        ]
        try:
            big_dqn_md  = biggest_dqn.to_markdown(index=False, floatfmt=".3f")
            big_ddqn_md = biggest_ddqn.to_markdown(index=False, floatfmt=".3f")
        except Exception:
            big_dqn_md  = biggest_dqn.to_string(index=False)
            big_ddqn_md = biggest_ddqn.to_string(index=False)
    else:
        avg_delta = float("nan")
        n_dqn_better = n_ddqn_better = n_tied = 0
        big_dqn_md = big_ddqn_md = "_no overlapping cells_"

    # ---- markdown report ----
    short_table_sr  = md_table_metric(df, "sr",       "short", fmt="{:.2f}")
    long_table_sr   = md_table_metric(df, "sr",       "long",  fmt="{:.2f}")
    short_table_pl  = md_table_metric(df, "path_len", "short", fmt="{:.2f}")
    long_table_pl   = md_table_metric(df, "path_len", "long",  fmt="{:.2f}")

    overall_verdict = (
        "DQN 整体略优" if (avg_delta == avg_delta and avg_delta >  0.02)
        else "DDQN 整体略优" if (avg_delta == avg_delta and avg_delta < -0.02)
        else "两者基本持平"
    )

    md = f"""# §4.5 cnn-dqn vs cnn-ddqn 全因子消融对比

> **不动 paper**：本报告纯属内部数据分析，不修改 `paper/main.tex`。
> 是否将论文 §4.5 的 base algo 由 cnn-ddqn 改为 cnn-dqn，由 Dr Sun 决定。

## 1. 实验设置

- **因子分解**：4 variants × 3 conditions × 2 distances = 24 cells（每个 base 一组，共 48 cells）
- **variants**：vanilla / +duel / +munch / +munch+duel
- **conditions**：full（AM+DQfD on）/ noAM / noDQfD
- **distances**：short（6–14 m）/ long（≥18 m）
- **runs per cell**：50，seed=110，goal_tolerance=0.3 m
- **筛选**：本表使用 SR mode（filter_all_succeed=false），全量 50 runs

### 数据来源
- **cnn-dqn**：`runs20260408_dqn/`（本日新跑，12 train + 24 infer，final checkpoint）
- **cnn-ddqn**：`runs20260408_ddqn/`（论文 §4.5 既有数据，多数 cells final ckpt，munch_duel 5 cells 用 MINTD）

### ⚠️ 不对称性 disclaim
DDQN 的 munch_duel × 5 cells 推理使用最小 TD loss checkpoint（`_MINTD` 后缀），其余 19 cells 与 DQN 一样使用 final checkpoint。该不对称源于既有论文实验的历史决定，本对比未消除该差异。

---

## 2. Success rate 对比表

格式：`DQN/DDQN (Δ)`，Δ = DQN − DDQN，正数表示 DQN 更高。

### 2.1 短距（6–14 m）

{short_table_sr}

### 2.2 长距（≥18 m）

{long_table_sr}

---

## 3. 平均路径长度对比表（单位 m）

格式：`DQN/DDQN (Δ)`，Δ = DQN − DDQN，正数表示 DQN 路径更长（更绕）。

### 3.1 短距

{short_table_pl}

### 3.2 长距

{long_table_pl}

---

## 4. 对比图

### 4.1 Success rate（短距）
![SR short](figs/dqn_vs_ddqn_sr_short.png)

### 4.2 Success rate（长距）
![SR long](figs/dqn_vs_ddqn_sr_long.png)

### 4.3 平均路径长度（短/长 双图）
![Path length](figs/dqn_vs_ddqn_pathlen.png)

---

## 5. 自动统计

- **24 cell 平均 ΔSR (DQN − DDQN)**：**{avg_delta:+.3f}**
- **DQN 更好** (ΔSR > +0.02)：**{n_dqn_better}** cells
- **DDQN 更好** (ΔSR < −0.02)：**{n_ddqn_better}** cells
- **打平** (|ΔSR| ≤ 0.02)：**{n_tied}** cells

### 5.1 DQN 优势最大的 3 cells

{big_dqn_md}

### 5.2 DDQN 优势最大的 3 cells

{big_ddqn_md}

---

## 6. 结论（待 Dr Sun 审阅）

- 平均 ΔSR = {avg_delta:+.3f}：**{overall_verdict}**
- DQN 优 / DDQN 优 / 打平 ≈ {n_dqn_better} / {n_ddqn_better} / {n_tied}
- **是否值得改论文 §4.5 用 cnn-dqn 替换 cnn-ddqn？** 需 Dr Sun 综合考虑：
  1. 整体 ΔSR 方向与显著性
  2. 路径质量（path_len, planning time）是否同步改善
  3. munch_duel cells 的不对称 checkpoint 选择是否需要重跑统一

> 数据：[raw_48.csv](raw_48.csv)
> 生成时间：{datetime.now().isoformat(timespec='seconds')}
"""

    md_file = out_dir / "ablation_20260408_dqn_vs_ddqn_base.md"
    md_file.write_text(md, encoding="utf-8")
    print(f"[compare] wrote {md_file}")
    print(f"[compare] wrote {fig_dir}/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
