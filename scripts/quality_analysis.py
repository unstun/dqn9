#!/usr/bin/env python3
"""质量模式分析：只看所有变体都成功的 runs，对比路径质量"""
import csv, os, glob, json

BASE = os.path.expanduser("~/DQN9/runs")
variants = ['cnn_ddqn','cnn_ddqn_duel','cnn_ddqn_mha','cnn_ddqn_md',
            'cnn_dqn_duel','cnn_dqn_mha','cnn_dqn_md']

def find_dir(variant, target_min):
    pat = os.path.join(BASE, f"abl_diag10k_kt02_infer_{variant}", "20260326_*")
    for d in sorted(glob.glob(pat), reverse=True):
        rj = os.path.join(d, "configs", "run.json")
        cp = os.path.join(d, "table2_kpis.csv")
        if not os.path.exists(rj) or not os.path.exists(cp):
            continue
        with open(rj) as f:
            cfg = json.load(f)
        args = cfg.get('args', cfg.get('infer', cfg))
        mc = args.get('rand_min_cost_m', 0)
        if abs(mc - target_min) < 0.5:
            return d
    return None

def load(dirpath):
    runs = {}
    with open(os.path.join(dirpath, "table2_kpis.csv")) as f:
        for row in csv.DictReader(f):
            ri = int(row['Run index'])
            s = float(row['Success rate']) >= 1.0
            pl = float(row['Average path length (m)'])
            cu = float(row['Average curvature (1/m)'])
            ct = float(row['Compute time (s)'])
            runs[ri] = (s, pl, cu, ct)
    return runs

for label, tmin in [('LONG', 18.0), ('SHORT', 6.0)]:
    print(f"\n{'='*60}")
    print(f"{label} — Quality Mode (0.3m retrained)")
    print(f"{'='*60}")

    data = {}
    for v in variants:
        d = find_dir(v, tmin)
        if d:
            data[v] = load(d)
            ns = sum(1 for _, (s, _, _, _) in data[v].items() if s)
            print(f"  {v:<20} succ={ns}/50  dir={os.path.basename(d)}")
        else:
            print(f"  {v:<20} MISSING")

    ss = {v: {ri for ri, (s, _, _, _) in r.items() if s} for v, r in data.items()}

    # DDQN 4 变体
    ddqn4 = [v for v in ['cnn_ddqn', 'cnn_ddqn_duel', 'cnn_ddqn_mha', 'cnn_ddqn_md'] if v in ss]
    if len(ddqn4) >= 2:
        inter = set.intersection(*[ss[v] for v in ddqn4])
        print(f"\n  DDQN {len(ddqn4)}变体交集: {len(inter)}/50")
        if inter:
            print(f"  {'Variant':<20} {'Path(m)':>8} {'Curv':>10} {'Time(s)':>8}")
            print(f"  {'-'*50}")
            res = []
            for v in ddqn4:
                vals = [(data[v][ri][1], data[v][ri][2], data[v][ri][3]) for ri in inter]
                n = len(vals)
                res.append((v, sum(x[0] for x in vals)/n, sum(x[1] for x in vals)/n, sum(x[2] for x in vals)/n))
            res.sort(key=lambda x: x[1])
            for v, pl, cu, ct in res:
                tag = " <-- MD" if v == "cnn_ddqn_md" else ""
                print(f"  {v:<20} {pl:>8.3f} {cu:>10.6f} {ct:>8.4f}{tag}")

    # 全变体交集
    if len(data) >= 3:
        all_inter = set.intersection(*ss.values())
        print(f"\n  全{len(data)}变体交集: {len(all_inter)}/50")
        if all_inter:
            print(f"  {'Variant':<20} {'Path(m)':>8} {'Curv':>10} {'Time(s)':>8}")
            print(f"  {'-'*50}")
            res = []
            for v in data:
                vals = [(data[v][ri][1], data[v][ri][2], data[v][ri][3]) for ri in all_inter]
                n = len(vals)
                res.append((v, sum(x[0] for x in vals)/n, sum(x[1] for x in vals)/n, sum(x[2] for x in vals)/n))
            res.sort(key=lambda x: x[1])
            for v, pl, cu, ct in res:
                tag = " <-- MD" if v == "cnn_ddqn_md" else ""
                print(f"  {v:<20} {pl:>8.3f} {cu:>10.6f} {ct:>8.4f}{tag}")

    # MD 两两对比
    if 'cnn_ddqn_md' in ss:
        print(f"\n  MD 两两质量对比:")
        for v2 in sorted(data.keys()):
            if v2 == 'cnn_ddqn_md':
                continue
            pair = ss['cnn_ddqn_md'] & ss[v2]
            if not pair:
                print(f"    vs {v2:<18} N=0 (无共同成功run)")
                continue
            md_p = sum(data['cnn_ddqn_md'][ri][1] for ri in pair) / len(pair)
            v2_p = sum(data[v2][ri][1] for ri in pair) / len(pair)
            md_c = sum(data['cnn_ddqn_md'][ri][2] for ri in pair) / len(pair)
            v2_c = sum(data[v2][ri][2] for ri in pair) / len(pair)
            winner = "MD" if md_p < v2_p else v2
            print(f"    vs {v2:<18} N={len(pair):>2}  MD={md_p:.2f}m/{md_c:.4f}  vs {v2_p:.2f}m/{v2_c:.4f}  -> {winner} wins path")
