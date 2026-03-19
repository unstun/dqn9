#!/usr/bin/env python3
"""聚合 AM x DQfD 组件消融推理结果：SR 模式 + Quality 模式"""
import csv
import json
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path("/home/sun/phdproject/dqn/DQN9/runs")

# 变体 → (推理目录后缀, 算法过滤)
VARIANTS = {
    "Full (AM+DQfD)": ("full",    None),
    "w/o DQfD":       ("noDQfD",  None),
    "w/o AM":         ("noAM",    None),
}


def parse_all():
    """解析所有推理 CSV，返回 {variant: {distance: [row_dict]}}"""
    data = {}

    for vname, (dir_suffix, algo_filter) in VARIANTS.items():
        infer_dir = RUNS_DIR / f"abl_amdqfd_infer_{dir_suffix}"
        if not infer_dir.exists():
            print(f"WARNING: {infer_dir} not found")
            continue

        data[vname] = {"long": [], "short": []}

        for ts_dir in sorted(infer_dir.iterdir()):
            if not ts_dir.is_dir() or not ts_dir.name.startswith("20"):
                continue

            run_json = ts_dir / "configs" / "run.json"
            if not run_json.exists():
                continue
            with open(run_json) as f:
                args = json.load(f)["args"]
            seed = args["seed"]
            profile = args["profile"]
            distance = "long" if "long" in profile else "short"

            kpi_csv = ts_dir / "table2_kpis.csv"
            if not kpi_csv.exists():
                continue
            with open(kpi_csv) as f:
                for row in csv.DictReader(f):
                    algo_name = row["Algorithm name"].strip()
                    if algo_filter and algo_name != algo_filter:
                        continue

                    entry = {
                        "seed": seed,
                        "run_idx": int(row["Run index"]),
                        "success": float(row["Success rate"]),
                        "path_length": float(row["Average path length (m)"]),
                        "curvature": float(row["Average curvature (1/m)"]),
                        "compute_time": float(row["Compute time (s)"]),
                    }
                    data[vname][distance].append(entry)

    return data


def sr_analysis(data):
    """SR 模式：每个变体/距离的成功率"""
    print("=" * 60)
    print("SR 模式 —— 成功率（50 runs per variant per distance）")
    print("=" * 60)

    for distance in ["long", "short"]:
        print(f"\n--- {distance.upper()} distance ---")
        print(f"{'Variant':<20} {'Runs':>5} {'Success':>8} {'SR%':>8}")
        print("-" * 45)
        results = []
        for vname in VARIANTS:
            rows = data.get(vname, {}).get(distance, [])
            n = len(rows)
            succ = sum(1 for r in rows if r["success"] >= 1.0)
            sr = succ / n * 100 if n > 0 else 0
            results.append((vname, n, succ, sr))
        results.sort(key=lambda x: -x[3])
        for vname, n, succ, sr in results:
            print(f"{vname:<20} {n:>5} {succ:>8} {sr:>7.1f}%")


def quality_analysis(data):
    """Quality 模式：筛选所有变体都成功的 runs，比较路径质量"""
    print("\n" + "=" * 60)
    print("Quality 模式 —— 全成功 runs 路径质量对比")
    print("=" * 60)

    for distance in ["long", "short"]:
        run_map = defaultdict(dict)
        for vname in VARIANTS:
            for entry in data.get(vname, {}).get(distance, []):
                key = (entry["seed"], entry["run_idx"])
                run_map[key][vname] = entry

        all_variants = set(VARIANTS.keys())
        all_succeed_keys = []
        for key, vdict in sorted(run_map.items()):
            if set(vdict.keys()) != all_variants:
                continue
            if all(vdict[v]["success"] >= 1.0 for v in all_variants):
                all_succeed_keys.append(key)

        n_total = sum(1 for k, v in run_map.items() if set(v.keys()) == all_variants)
        n_quality = len(all_succeed_keys)

        print(f"\n--- {distance.upper()} distance ---")
        print(f"总共 {n_total} 个完整 runs，其中 {n_quality} 个所有变体都成功")

        if n_quality == 0:
            print("（无符合条件的 runs）")
            continue

        print(f"\n{'Variant':<20} {'N':>4} {'AvgLen(m)':>10} {'AvgCurv':>10} {'AvgTime(s)':>11}")
        print("-" * 60)
        results = []
        for vname in VARIANTS:
            lengths = []
            curvatures = []
            times = []
            for key in all_succeed_keys:
                e = run_map[key][vname]
                lengths.append(e["path_length"])
                curvatures.append(e["curvature"])
                times.append(e["compute_time"])
            avg_len = sum(lengths) / len(lengths)
            avg_curv = sum(curvatures) / len(curvatures)
            avg_time = sum(times) / len(times)
            results.append((vname, n_quality, avg_len, avg_curv, avg_time))

        results.sort(key=lambda x: x[2])
        for vname, n, al, ac, at in results:
            print(f"{vname:<20} {n:>4} {al:>10.3f} {ac:>10.6f} {at:>11.5f}")


def main():
    print("正在解析 AM x DQfD 消融推理结果...\n")
    data = parse_all()

    print("数据完整性检查：")
    for vname in VARIANTS:
        for dist in ["long", "short"]:
            n = len(data.get(vname, {}).get(dist, []))
            status = "OK" if n == 50 else f"({n}/50)"
            print(f"  {vname:<20} {dist:<6} {n:>3} runs {status}")

    sr_analysis(data)
    quality_analysis(data)


if __name__ == "__main__":
    main()
