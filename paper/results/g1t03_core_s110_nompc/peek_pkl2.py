import pickle
from pathlib import Path
HERE = Path(__file__).parent
data = pickle.load(open(HERE / "long" / "paths_for_plot.pkl", "rb"))
paths = data["paths"]
# 看一条路径的内部结构
k = ('realmap_a', 0, 'CNN-DDQN+Duel')
v = paths[k]
for sk, sv in v.items():
    print(f"  {sk}: type={type(sv)}", end="")
    if hasattr(sv, 'shape'): print(f", shape={sv.shape}, dtype={sv.dtype}")
    elif hasattr(sv, '__len__'): print(f", len={len(sv)}")
    else: print(f", val={sv}")

# 找共同成功的 run
import pandas as pd
df = pd.read_csv(HERE / "long" / "table2_kpis_raw.csv")
algos = df["Algorithm"].unique()
print(f"\nAlgos: {algos}")
success_runs = []
for rid in range(50):
    all_ok = True
    for algo in algos:
        row = df[(df["run_idx"]==rid) & (df["Algorithm"]==algo)]
        if row.empty or row.iloc[0]["success_rate"] < 1.0:
            all_ok = False; break
    if all_ok:
        success_runs.append(rid)
print(f"Common-success runs (long): {success_runs[:10]}... total={len(success_runs)}")

# 检查路径 key 是否存在
for rid in success_runs[:3]:
    for a in ["CNN-DDQN+Duel", "Hybrid A* (Dang 2022)", "Hybrid A*", "RRT* (Yoon 2018)", "RRT*"]:
        k = ('realmap_a', rid, a)
        if k in paths:
            print(f"  run={rid}, algo={a}: EXISTS, keys={list(paths[k].keys())}")
