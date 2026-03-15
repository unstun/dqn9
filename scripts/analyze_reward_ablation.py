#!/usr/bin/env python3
"""Analyze reward ablation results: 18 experiments × 2 modes vs baseline.

Usage:  python scripts/analyze_reward_ablation.py [--runs-root runs]
Output: runs/reward_ablation_summary/ with CSV tables and markdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

EXPERIMENTS = [
    "T1a_kt02", "T1b_kt04", "T1c_kt08",
    "T2a_smooth_mild", "T2b_smooth_mid", "T2c_smooth_aggr",
    "T3a_clip8", "T3b_clip12", "T3c_clip20",
    "T4a_obs_mild", "T4b_obs_mid", "T4c_obs_aggr",
    "T5a_eff03", "T5b_eff08", "T5c_eff15",
    "C1_time_smooth_clip", "C2_plus_obs", "C3_full",
]

APPROACH_LABELS = {
    "T1": "① Time penalty (k_t)",
    "T2": "② Smoothness ↓",
    "T3": "③ Reward clip ↑",
    "T4": "④ Obstacle ↓",
    "T5": "⑤ Efficiency penalty",
    "C": "Combination",
}


def read_kpis(kpi_path: Path) -> list[dict]:
    rows = []
    with open(kpi_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_by_algo(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-run rows into mean stats by algorithm."""
    algo_data: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        algo = row.get("Algorithm name", row.get("Algorithm", ""))
        algo_data[algo].append(row)

    result = {}
    for algo, runs in algo_data.items():
        n = len(runs)
        success = sum(1 for r in runs if float(r.get("Success rate", r.get("success_rate", 0))) >= 1.0)
        sr = success / n if n > 0 else 0.0

        def mean_col(*keys):
            vals = []
            for r in runs:
                if float(r.get("Success rate", r.get("success_rate", 0))) < 1.0:
                    continue  # skip failed runs for quality metrics
                for k in keys:
                    if k in r:
                        vals.append(float(r[k]))
                        break
            return sum(vals) / len(vals) if vals else 0.0

        result[algo] = {
            "n_runs": n,
            "n_success": success,
            "sr": sr,
            "path_m": mean_col("Average path length (m)", "avg_path_length_m"),
            "curv": mean_col("Average curvature (1/m)", "avg_curvature_1_m"),
            "plan_time": mean_col("Planning time (s)", "planning_time_s"),
        }
    return result


def find_latest_run(runs_root: Path, exp: str, mode: str) -> Path | None:
    infer_out = runs_root / f"reward_abl_infer_{exp}"
    if not infer_out.exists():
        return None
    run_dirs = sorted(infer_out.glob("20*"), reverse=True)
    for rd in run_dirs:
        kpi = rd / "table2_kpis.csv"
        rj = rd / "configs" / "run.json"
        if not kpi.exists():
            continue
        if rj.exists():
            with open(rj) as f:
                rjd = json.load(f)
            min_cost = rjd.get("args", {}).get("rand_min_cost_m", 0)
            if mode == "sr_long" and min_cost >= 18.0:
                return rd
            elif mode == "sr_short" and min_cost < 18.0:
                return rd
        else:
            return rd
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--baseline-dir", type=str, default="")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = runs_root / "reward_ablation_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(
        "paperruns/ablation_diag10k_minlossvsbaseline2/cnn_ddqn_duel"
    )

    MODES = ["sr_long", "sr_short"]
    all_results = []  # (exp, mode, drl_stats, baseline_stats)

    # Baseline
    for mode in MODES:
        kpi_file = baseline_dir / mode / "table2_kpis.csv"
        if kpi_file.exists():
            agg = aggregate_by_algo(read_kpis(kpi_file))
            drl = None
            baselines = {}
            for algo, stats in agg.items():
                if "CNN" in algo:
                    drl = stats
                elif "RRT" in algo:
                    baselines["RRT*"] = stats
                elif "LO" in algo or "HA" in algo:
                    baselines["LO-HA*"] = stats
            if drl:
                all_results.append(("baseline", mode, drl, baselines))

    # Experiments
    for exp in EXPERIMENTS:
        for mode in MODES:
            rd = find_latest_run(runs_root, exp, mode)
            if rd is None:
                print(f"  MISS: {exp}/{mode}")
                continue
            kpi_file = rd / "table2_kpis.csv"
            if not kpi_file.exists():
                continue
            agg = aggregate_by_algo(read_kpis(kpi_file))
            drl = None
            baselines = {}
            for algo, stats in agg.items():
                if "CNN" in algo:
                    drl = stats
                elif "RRT" in algo:
                    baselines["RRT*"] = stats
                elif "LO" in algo or "HA" in algo:
                    baselines["LO-HA*"] = stats
            if drl:
                all_results.append((exp, mode, drl, baselines))

    # Build summary rows
    summary_rows = []
    for exp, mode, drl, baselines in all_results:
        loha = baselines.get("LO-HA*", {})
        loha_pl = loha.get("path_m", 0)
        rrt = baselines.get("RRT*", {})
        rrt_pl = rrt.get("path_m", 0)

        gap_loha = (drl["path_m"] - loha_pl) if loha_pl > 0 else float("nan")
        gap_pct = (gap_loha / loha_pl * 100) if loha_pl > 0 else float("nan")

        # Approach label
        prefix = exp[:2] if exp != "baseline" else ""
        approach = APPROACH_LABELS.get(prefix, exp)

        summary_rows.append({
            "experiment": exp,
            "approach": approach,
            "mode": mode,
            "SR": f"{drl['sr']:.0%}",
            "SR_raw": drl["sr"],
            "n_runs": drl["n_runs"],
            "n_success": drl["n_success"],
            "path_m": f"{drl['path_m']:.3f}",
            "path_m_raw": drl["path_m"],
            "curv": f"{drl['curv']:.4f}",
            "curv_raw": drl["curv"],
            "plan_s": f"{drl['plan_time']:.4f}",
            "LOHA_path": f"{loha_pl:.3f}" if loha_pl else "N/A",
            "RRT_path": f"{rrt_pl:.3f}" if rrt_pl else "N/A",
            "gap_m": f"{gap_loha:+.3f}" if gap_loha == gap_loha else "N/A",
            "gap_pct": f"{gap_pct:+.1f}%" if gap_pct == gap_pct else "N/A",
            "gap_raw": gap_loha if gap_loha == gap_loha else 999,
        })

    # Write CSV
    csv_path = out_dir / "reward_ablation_comparison.csv"
    if summary_rows:
        cols = ["experiment", "approach", "mode", "SR", "n_runs", "n_success",
                "path_m", "curv", "plan_s", "LOHA_path", "RRT_path", "gap_m", "gap_pct"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Written: {csv_path}")

    # Write markdown
    md_path = out_dir / "reward_ablation_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Reward Ablation: CNN-DDQN+Duel Path Length Comparison\n\n")
        f.write(f"18 experiments + baseline, 50 runs each, realmap_a, diag EDT\n\n")

        for mode in MODES:
            f.write(f"\n## {mode}\n\n")
            f.write("| Experiment | Approach | SR | Path (m) | Curv | Plan (s) | vs LO-HA* | Gap% |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            mode_rows = sorted(
                [r for r in summary_rows if r["mode"] == mode],
                key=lambda r: r["gap_raw"]
            )
            for row in mode_rows:
                marker = " **★**" if row["experiment"] == "baseline" else ""
                f.write(
                    f"| {row['experiment']}{marker} | {row['approach']} | {row['SR']} | "
                    f"{row['path_m']} | {row['curv']} | {row['plan_s']} | "
                    f"{row['gap_m']} | {row['gap_pct']} |\n"
                )

        # Best per approach
        f.write("\n## Best per approach (smallest path gap vs LO-HA*)\n\n")
        for mode in MODES:
            f.write(f"\n### {mode}\n\n")
            mode_rows = [r for r in summary_rows if r["mode"] == mode and r["experiment"] != "baseline"]
            mode_rows.sort(key=lambda r: r["gap_raw"])
            for i, r in enumerate(mode_rows[:5]):
                bl = [x for x in summary_rows if x["experiment"] == "baseline" and x["mode"] == mode]
                bl_gap = bl[0]["gap_m"] if bl else "?"
                f.write(
                    f"{i+1}. **{r['experiment']}**: path={r['path_m']}m, gap={r['gap_m']}m ({r['gap_pct']}), "
                    f"SR={r['SR']}, curv={r['curv']} (baseline gap={bl_gap})\n"
                )

        # Summary
        f.write("\n## Key findings\n\n")
        for mode in MODES:
            bl = [r for r in summary_rows if r["experiment"] == "baseline" and r["mode"] == mode]
            best = min(
                [r for r in summary_rows if r["mode"] == mode and r["experiment"] != "baseline"],
                key=lambda r: r["gap_raw"],
                default=None,
            )
            if bl and best:
                improvement = bl[0]["gap_raw"] - best["gap_raw"]
                f.write(
                    f"- **{mode}**: baseline gap={bl[0]['gap_m']}m → best={best['experiment']} "
                    f"gap={best['gap_m']}m (improvement: {improvement:+.3f}m)\n"
                )

    print(f"Written: {md_path}")
    print(f"\nTotal: {len(summary_rows)} rows ({len(set(r['experiment'] for r in summary_rows))} experiments)")


if __name__ == "__main__":
    main()
