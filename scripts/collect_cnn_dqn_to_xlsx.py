#!/usr/bin/env python3
# ============================================================================
# collect_cnn_dqn_to_xlsx.py
# ----------------------------------------------------------------------------
# Aggregate the 24 cnn-dqn inference outputs of the §4.5 full-factor ablation
# into plot-ready xlsx workbooks. Each infer output directory contains:
#
#   <RUNS_ROOT>/<out>/<YYYYMMDD_HHMMSS>/
#     table2_kpis.csv          (raw per-run KPIs; one row per run)
#     table2_kpis_mean.csv     (aggregated mean KPIs; one row per algo)
#     paths_all.csv            (all trajectory points; for figure plotting)
#     ...
#
# The 24 inference configs factor as 4 variants x 3 conditions x 2 distances:
#   variants   = vanilla / duel / munch / munch_duel
#   conditions = full / noAM / noDQfD
#   distances  = short (6-14 m) / long (>= 18 m)
#
# Output workbooks written to --out-dir:
#
#   cnn_dqn_45_summary.xlsx
#     sr_table            24 rows, (variant,condition,distance) + key KPIs
#     sr_short_pivot      4 x 3 pivot: rows=variant, cols=condition, SR short
#     sr_long_pivot       same, long
#     pathlen_short_pivot same, path length short
#     pathlen_long_pivot  same, path length long
#     meta                pipeline meta + missing-dir log
#
#   cnn_dqn_45_raw_kpi.xlsx
#     24 sheets, one per (variant_condition_distance), full per-run KPI rows
#
#   cnn_dqn_45_paths.xlsx
#     paths_short         all short-distance trajectory points (flat, tagged)
#     paths_long          all long-distance trajectory points  (flat, tagged)
#
# Usage:
#   python scripts/collect_cnn_dqn_to_xlsx.py \
#       --runs-root /home/ubuntu/DQN9/runs20260408_dqn \
#       --out-dir   /home/ubuntu/DQN9/runs20260408_dqn/aggregated_<ts>
# ============================================================================

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------------
# 24-cell factor table (variant x condition x distance -> out_dir_name)
# ----------------------------------------------------------------------------
VARIANTS = [
    ("",            "vanilla"),      # cnn-dqn base
    ("_duel",       "duel"),         # + dueling
    ("_munch",      "munch"),        # + munchausen
    ("_munch_duel", "munch_duel"),   # + munchausen + dueling
]
CONDITIONS = ["full", "noAM", "noDQfD"]
DISTANCES = ["short", "long"]


def enumerate_infer_cells():
    """Yield one descriptor dict per expected infer output directory."""
    for v_suffix, v_label in VARIANTS:
        for cond in CONDITIONS:
            for dist in DISTANCES:
                out_dir_name = (
                    f"abl_amdqfd_cnn_dqn{v_suffix}_infer_{cond}_sr_{dist}"
                )
                yield {
                    "variant":   v_label,
                    "condition": cond,
                    "distance":  dist,
                    "out_dir":   out_dir_name,
                }


def latest_timestamp_subdir(base: Path) -> Path | None:
    """Return the most recent <YYYYMMDD_HHMMSS> subdir, or None."""
    if not base.exists():
        return None
    cands = [
        p for p in base.iterdir()
        if p.is_dir() and len(p.name) >= 8 and p.name[:8].isdigit()
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: p.name)


# ----------------------------------------------------------------------------
# Collectors
# ----------------------------------------------------------------------------
def collect_one_cell(runs_root: Path, cell: dict) -> dict:
    """Read the 3 CSVs (if present) for one infer cell."""
    base = runs_root / cell["out_dir"]
    ts_dir = latest_timestamp_subdir(base)
    result = {
        "cell": cell,
        "ts_dir": ts_dir,
        "sr_row": None,
        "raw_df": None,
        "paths_df": None,
        "present": False,
    }
    if ts_dir is None:
        return result
    result["present"] = True

    mean_csv = ts_dir / "table2_kpis_mean.csv"
    raw_csv = ts_dir / "table2_kpis.csv"
    paths_csv = ts_dir / "paths_all.csv"

    # ---- table2_kpis_mean.csv (one row per algo; cnn-dqn configs run 1 algo) ----
    if mean_csv.exists():
        try:
            mdf = pd.read_csv(mean_csv)
            if len(mdf) > 0:
                row = mdf.iloc[0]
                result["sr_row"] = {
                    "variant":       cell["variant"],
                    "condition":     cell["condition"],
                    "distance":      cell["distance"],
                    "out_dir":       cell["out_dir"],
                    "algorithm":     row.get("Algorithm name", ""),
                    "environment":   row.get("Environment", ""),
                    "success_rate":  row.get("Success rate", float("nan")),
                    "path_length_m": row.get("Average path length (m)", float("nan")),
                    "path_time_s":   row.get("Path time (s)", float("nan")),
                    "curvature_1pm": row.get("Average curvature (1/m)", float("nan")),
                    "planning_time_s": row.get("Planning time (s)", float("nan")),
                    "compute_time_s":  row.get("Compute time (s)", float("nan")),
                    "corners":       row.get("Number of path corners", float("nan")),
                    "composite":     row.get("Composite score", float("nan")),
                }
        except Exception as exc:
            print(f"[collect] WARN failed to read {mean_csv}: {exc}")

    # ---- table2_kpis.csv (raw per-run) ----
    if raw_csv.exists():
        try:
            rdf = pd.read_csv(raw_csv)
            rdf.insert(0, "distance", cell["distance"])
            rdf.insert(0, "condition", cell["condition"])
            rdf.insert(0, "variant", cell["variant"])
            result["raw_df"] = rdf
        except Exception as exc:
            print(f"[collect] WARN failed to read {raw_csv}: {exc}")

    # ---- paths_all.csv (flat trajectory points) ----
    if paths_csv.exists():
        try:
            pdf = pd.read_csv(paths_csv)
            pdf.insert(0, "distance", cell["distance"])
            pdf.insert(0, "condition", cell["condition"])
            pdf.insert(0, "variant", cell["variant"])
            result["paths_df"] = pdf
        except Exception as exc:
            print(f"[collect] WARN failed to read {paths_csv}: {exc}")

    return result


def short_sheet_name(variant: str, condition: str, distance: str) -> str:
    """Excel sheet names must be <= 31 chars; our longest fits naturally."""
    name = f"{variant}_{condition}_{distance}"
    return name[:31]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate 24 cnn-dqn inference outputs into xlsx workbooks"
    )
    ap.add_argument("--runs-root", required=True, type=Path,
                    help="e.g. /home/ubuntu/DQN9/runs20260408_dqn")
    ap.add_argument("--out-dir",   required=True, type=Path,
                    help="Output directory for the 3 xlsx workbooks")
    args = ap.parse_args()

    runs_root: Path = args.runs_root.resolve()
    out_dir:   Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[collect] runs_root = {runs_root}")
    print(f"[collect] out_dir   = {out_dir}")

    cells = list(enumerate_infer_cells())
    assert len(cells) == 24, f"expected 24 cells, got {len(cells)}"
    print(f"[collect] expecting {len(cells)} infer output dirs")

    sr_rows, raw_sheets, paths_frames, missing = [], {}, [], []
    for cell in cells:
        res = collect_one_cell(runs_root, cell)
        if not res["present"]:
            missing.append(cell["out_dir"])
            print(f"[collect] MISSING {cell['out_dir']}")
            continue
        if res["sr_row"]:
            sr_rows.append(res["sr_row"])
        if res["raw_df"] is not None:
            sheet = short_sheet_name(cell["variant"], cell["condition"], cell["distance"])
            raw_sheets[sheet] = res["raw_df"]
        if res["paths_df"] is not None:
            paths_frames.append(res["paths_df"])

    # ------------------------------------------------------------------
    # Build summary DataFrames
    # ------------------------------------------------------------------
    sr_df = pd.DataFrame(sr_rows)
    if not sr_df.empty:
        sr_df = sr_df.sort_values(
            ["condition", "distance", "variant"]
        ).reset_index(drop=True)

    def pivot(metric: str, distance: str) -> pd.DataFrame:
        if sr_df.empty:
            return pd.DataFrame()
        sub = sr_df[sr_df["distance"] == distance]
        if sub.empty:
            return pd.DataFrame()
        return sub.pivot_table(
            index="variant", columns="condition", values=metric,
            aggfunc="first",
        ).reindex(index=[v[1] for v in VARIANTS], columns=CONDITIONS)

    sr_short_pivot     = pivot("success_rate",  "short")
    sr_long_pivot      = pivot("success_rate",  "long")
    pathlen_short_pivot = pivot("path_length_m", "short")
    pathlen_long_pivot  = pivot("path_length_m", "long")

    meta_df = pd.DataFrame([
        {"key": "generated_at",     "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "runs_root",        "value": str(runs_root)},
        {"key": "cells_expected",   "value": len(cells)},
        {"key": "cells_found",      "value": len(sr_rows)},
        {"key": "cells_missing",    "value": len(missing)},
        {"key": "missing_out_dirs", "value": ",".join(missing) if missing else "-"},
    ])

    # ------------------------------------------------------------------
    # 1. cnn_dqn_45_summary.xlsx
    # ------------------------------------------------------------------
    summary_xlsx = out_dir / "cnn_dqn_45_summary.xlsx"
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as w:
        (sr_df if not sr_df.empty else pd.DataFrame({"_": ["no data"]})) \
            .to_excel(w, sheet_name="sr_table", index=False)
        sr_short_pivot.to_excel(w, sheet_name="sr_short_pivot")
        sr_long_pivot.to_excel(w,  sheet_name="sr_long_pivot")
        pathlen_short_pivot.to_excel(w, sheet_name="pathlen_short_pivot")
        pathlen_long_pivot.to_excel(w,  sheet_name="pathlen_long_pivot")
        meta_df.to_excel(w, sheet_name="meta", index=False)
    print(f"[collect] wrote {summary_xlsx}")

    # ------------------------------------------------------------------
    # 2. cnn_dqn_45_raw_kpi.xlsx  (24 sheets)
    # ------------------------------------------------------------------
    raw_xlsx = out_dir / "cnn_dqn_45_raw_kpi.xlsx"
    with pd.ExcelWriter(raw_xlsx, engine="openpyxl") as w:
        if not raw_sheets:
            pd.DataFrame({"_": ["no data"]}).to_excel(w, sheet_name="empty", index=False)
        else:
            for sheet, df in raw_sheets.items():
                df.to_excel(w, sheet_name=sheet, index=False)
    print(f"[collect] wrote {raw_xlsx}  ({len(raw_sheets)} sheets)")

    # ------------------------------------------------------------------
    # 3. cnn_dqn_45_paths.xlsx  (short / long)
    # ------------------------------------------------------------------
    if paths_frames:
        all_paths = pd.concat(paths_frames, ignore_index=True)
        short_paths = all_paths[all_paths["distance"] == "short"].reset_index(drop=True)
        long_paths  = all_paths[all_paths["distance"] == "long"].reset_index(drop=True)
    else:
        short_paths = long_paths = pd.DataFrame(
            columns=["variant", "condition", "distance",
                     "env", "run_idx", "algo", "point_idx", "x_m", "y_m", "success"]
        )

    paths_xlsx = out_dir / "cnn_dqn_45_paths.xlsx"
    with pd.ExcelWriter(paths_xlsx, engine="openpyxl") as w:
        short_paths.to_excel(w, sheet_name="paths_short", index=False)
        long_paths.to_excel(w,  sheet_name="paths_long",  index=False)
    print(
        f"[collect] wrote {paths_xlsx}  "
        f"(short_rows={len(short_paths)}  long_rows={len(long_paths)})"
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print(
        f"[collect] DONE.  cells_found={len(sr_rows)}/{len(cells)}  "
        f"missing={len(missing)}"
    )
    if missing:
        for m in missing:
            print(f"[collect]   missing: {m}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
