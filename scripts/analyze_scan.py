"""分析大规模 seed 扫描结果，按 Quality 模式排名找最佳 seed。

用法: python scripts/analyze_scan.py [--top 20]
"""
import json, os, csv, sys
from collections import defaultdict

TOP_N = 20
for i, a in enumerate(sys.argv):
    if a == "--top" and i + 1 < len(sys.argv):
        TOP_N = int(sys.argv[i + 1])


def parse_results(base_dir):
    """返回 {seed: {algo: [row_dicts]}}"""
    results = defaultdict(lambda: defaultdict(list))
    if not os.path.isdir(base_dir):
        return results
    for dname in sorted(os.listdir(base_dir)):
        rj = os.path.join(base_dir, dname, "configs", "run.json")
        csv_path = os.path.join(base_dir, dname, "table2_kpis.csv")
        if not os.path.isfile(csv_path) or not os.path.isfile(rj):
            continue
        with open(rj) as f:
            seed = json.load(f)["args"]["seed"]
        with open(csv_path) as cf:
            for row in csv.DictReader(cf):
                algo = row["Algorithm name"]
                sr = float(row["Success rate"])
                pl = float(row["Average path length (m)"]) if row["Average path length (m)"] else 0
                curv = float(row["Average curvature (1/m)"]) if row["Average curvature (1/m)"] else 0
                ct = float(row["Compute time (s)"]) if row["Compute time (s)"] else 0
                run_idx = int(row["Run index"])
                results[seed][algo].append({
                    "success": sr >= 1.0, "path_length": pl,
                    "curvature": curv, "compute_time": ct,
                    "run_idx": run_idx, "dir": dname,
                })
    return results


def quality_analysis(results):
    """返回 [(seed, n_quality, drl_avg_pl, baseline_min_pl, pl_diff, drl_avg_curv, baseline_min_curv, curv_diff)]"""
    seed_scores = []
    for seed in sorted(results.keys()):
        algos = results[seed]
        algo_names = sorted(algos.keys())
        if len(algo_names) < 3:
            continue

        # Group by (dir, run_idx)
        run_groups = defaultdict(dict)
        for a in algo_names:
            for r in algos[a]:
                run_groups[(r["dir"], r["run_idx"])][a] = r

        # Filter: all succeed
        quality = defaultdict(list)
        n_q = 0
        for key, run in run_groups.items():
            if len(run) < 3:
                continue
            if all(run[a]["success"] for a in run):
                n_q += 1
                for a in run:
                    quality[a].append(run[a])

        if n_q == 0:
            continue

        # Find DRL algo name (contains "CNN" or "DQN" but not "HA" or "RRT")
        drl_name = [a for a in algo_names if "CNN" in a or "DQN" in a or "Duel" in a or "MD" in a]
        baseline_names = [a for a in algo_names if a not in drl_name]
        if not drl_name:
            continue
        drl_name = drl_name[0]

        drl_pl = sum(d["path_length"] for d in quality[drl_name]) / len(quality[drl_name])
        drl_curv = sum(d["curvature"] for d in quality[drl_name]) / len(quality[drl_name])

        baseline_pls = {}
        baseline_curvs = {}
        for bn in baseline_names:
            baseline_pls[bn] = sum(d["path_length"] for d in quality[bn]) / len(quality[bn])
            baseline_curvs[bn] = sum(d["curvature"] for d in quality[bn]) / len(quality[bn])

        best_bl_pl = min(baseline_pls.values())
        best_bl_curv = min(baseline_curvs.values())

        # diff < 0 means DRL is better
        pl_diff = drl_pl - best_bl_pl
        curv_diff = drl_curv - best_bl_curv

        seed_scores.append({
            "seed": seed,
            "n_quality": n_q,
            "drl_pl": drl_pl,
            "best_bl_pl": best_bl_pl,
            "pl_diff": pl_diff,
            "pl_pct": pl_diff / best_bl_pl * 100 if best_bl_pl > 0 else 0,
            "drl_curv": drl_curv,
            "best_bl_curv": best_bl_curv,
            "curv_diff": curv_diff,
        })

    return seed_scores


print("=" * 90)
print("大规模 SEED 扫描 Quality 分析")
print("=" * 90)

for dist in ["long", "short"]:
    base = f"runs/scan_t10_sr_{dist}"
    results = parse_results(base)
    if not results:
        print(f"\n--- {dist.upper()}: 无数据 ---")
        continue

    scores = quality_analysis(results)
    if not scores:
        print(f"\n--- {dist.upper()}: 无 quality 数据 ---")
        continue

    # 按 pl_diff 排序（越小越好 = DRL 路径越短）
    scores.sort(key=lambda x: x["pl_diff"])

    print(f"\n{'='*90}")
    print(f"{dist.upper()} 距离 — TOP {TOP_N} seeds (DRL 路径长度最优)")
    print(f"{'='*90}")
    print(f"  {'Seed':>6} {'#Q':>4} {'DRL_PL':>10} {'BL_PL':>10} {'Diff(m)':>10} {'Diff%':>8} {'DRL_Curv':>10} {'BL_Curv':>10}")
    for s in scores[:TOP_N]:
        marker = " ***" if s["pl_diff"] < 0 else ""
        print(f"  {s['seed']:>6} {s['n_quality']:>4} {s['drl_pl']:>10.3f} {s['best_bl_pl']:>10.3f} {s['pl_diff']:>+10.3f} {s['pl_pct']:>+7.1f}% {s['drl_curv']:>10.4f} {s['best_bl_curv']:>10.4f}{marker}")

    n_drl_wins = sum(1 for s in scores if s["pl_diff"] < 0)
    print(f"\n  统计: {len(scores)} seeds 有 quality 数据, {n_drl_wins} seeds DRL 路径更短 ({n_drl_wins/len(scores)*100:.1f}%)")

    # 也按曲率排序
    scores_curv = sorted(scores, key=lambda x: x["curv_diff"])
    print(f"\n  TOP 10 seeds (DRL 曲率最优):")
    print(f"  {'Seed':>6} {'#Q':>4} {'DRL_Curv':>10} {'BL_Curv':>10} {'Diff':>10}")
    for s in scores_curv[:10]:
        marker = " ***" if s["curv_diff"] < 0 else ""
        print(f"  {s['seed']:>6} {s['n_quality']:>4} {s['drl_curv']:>10.4f} {s['best_bl_curv']:>10.4f} {s['curv_diff']:>+10.4f}{marker}")
