import pickle
from pathlib import Path
HERE = Path(__file__).parent
for suite in ["long", "short"]:
    data = pickle.load(open(HERE / suite / "paths_for_plot.pkl", "rb"))
    paths = data["paths"]
    keys = sorted(paths.keys(), key=lambda x: (x[1], x[2]))
    print(f"\n=== {suite} ({len(keys)} keys) ===")
    runs = sorted(set(k[1] for k in keys))
    for rid in runs:
        status = []
        for a in ["CNN-DDQN+Duel", "Hybrid A*", "RRT*"]:
            k = ("realmap_a", rid, a)
            if k in paths:
                s = paths[k]["success"]
                n = len(paths[k]["xy_cells"])
                status.append(f"{a[:8]}:{'OK' if s else 'FAIL'}({n}pts)")
        print(f"  run {rid}: {', '.join(status)}")
