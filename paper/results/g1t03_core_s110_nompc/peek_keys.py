import pickle
from pathlib import Path
HERE = Path(__file__).parent
data = pickle.load(open(HERE / "long" / "paths_for_plot.pkl", "rb"))
paths = data["paths"]
# 列出所有 key
keys = sorted(paths.keys(), key=lambda x: (x[0], x[1], x[2]))
for k in keys[:30]:
    print(k, "success=", paths[k].get("success"), "len=", len(paths[k].get("xy_cells",[])))
print(f"... total keys: {len(keys)}")

# 检查 run 10 的 key
print("\n--- run 10 keys ---")
for k in keys:
    if k[1] == 10:
        print(k)
