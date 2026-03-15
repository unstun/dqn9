#!/usr/bin/env python3
"""Aggregate MPC comparison chunk results into unified CSV/MD reports.

Usage:  python scripts/aggregate_mpc_chunks.py [--runs-root runs]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROFILES = [
    "reward_abl_mpc_kt02_sr_long",
    "reward_abl_mpc_kt02_sr_short",
]
CHUNKS = 7


def find_latest_run_dir(runs_root: Path, exp_name: str) -> Path | None:
    exp_dir = runs_root / exp_name
    if not exp_dir.exists():
        return None
    run_dirs = sorted(exp_dir.glob("20*"), reverse=True)
    for rd in run_dirs:
        if (rd / "table2_kpis.csv").exists():
            return rd
    return None


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="runs")
    args = ap.parse_args()
    runs_root = Path(args.runs_root)

    out_dir = runs_root / "mpc_comparison_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    for profile in PROFILES:
        print(f"\n=== {profile} ===")
        all_rows: list[dict] = []
        chunks_found = 0

        for chunk_idx in range(CHUNKS):
            chunk_name = f"{profile}_chunk{chunk_idx}"
            rd = find_latest_run_dir(runs_root, chunk_name)
            if rd is None:
                print(f"  WARN: {chunk_name} not found")
                continue
            rows = read_csv(rd / "table2_kpis.csv")
            all_rows.extend(rows)
            chunks_found += 1
            print(f"  ✓ {chunk_name}: {len(rows)} rows")

        if not all_rows:
            print(f"  No data for {profile}")
            continue

        print(f"  Total: {len(all_rows)} rows from {chunks_found}/{CHUNKS} chunks")

        # Save combined per-run CSV
        combined_csv = out_dir / f"{profile}_all_runs.csv"
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved: {combined_csv}")

        # Compute per-algorithm mean stats
        algo_data: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            algo = row.get("Algorithm name", row.get("Algorithm", "unknown"))
            algo_data[algo].append(row)

        mean_rows = []
        for algo, rows in sorted(algo_data.items()):
            n = len(rows)
            success = sum(
                1 for r in rows
                if float(r.get("Success rate", r.get("success_rate", 0))) >= 1.0
            )
            sr = success / n if n > 0 else 0.0

            def safe_mean(key_candidates):
                vals = []
                for r in rows:
                    if float(r.get("Success rate", r.get("success_rate", 0))) < 1.0:
                        continue
                    for k in key_candidates:
                        if k in r and r[k]:
                            try:
                                vals.append(float(r[k]))
                            except (ValueError, TypeError):
                                pass
                            break
                return mean(vals) if vals else float("nan")

            mean_rows.append({
                "Algorithm": algo,
                "N_runs": n,
                "N_success": success,
                "SR": f"{sr:.0%}",
                "Path_m": f"{safe_mean(['Average path length (m)', 'avg_path_length_m']):.3f}",
                "Curvature": f"{safe_mean(['Average curvature (1/m)', 'avg_curvature_1_m']):.4f}",
                "Plan_time_s": f"{safe_mean(['Planning time (s)', 'planning_time_s']):.4f}",
            })

        # Save mean summary
        mean_csv = out_dir / f"{profile}_mean.csv"
        with open(mean_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mean_rows[0].keys())
            writer.writeheader()
            writer.writerows(mean_rows)

        # Print markdown table
        print(f"\n  ### {profile} — Mean Summary")
        print(f"  | Algorithm | Runs | Success | SR | Path(m) | Curv | Plan(s) |")
        print(f"  |-----------|------|---------|-----|---------|------|---------|")
        for r in mean_rows:
            print(f"  | {r['Algorithm']} | {r['N_runs']} | {r['N_success']} | {r['SR']} | {r['Path_m']} | {r['Curvature']} | {r['Plan_time_s']} |")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
