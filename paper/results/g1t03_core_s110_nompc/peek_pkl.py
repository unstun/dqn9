import pickle, pprint
from pathlib import Path
HERE = Path(__file__).parent
for suite in ["long", "short"]:
    p = HERE / suite / "paths_for_plot.pkl"
    data = pickle.load(open(p, "rb"))
    print(f"\n=== {suite} ===")
    print(f"type={type(data)}, len={len(data) if hasattr(data,'__len__') else 'N/A'}")
    if isinstance(data, dict):
        for k in list(data.keys())[:5]:
            v = data[k]
            print(f"  key={k!r}, type={type(v)}, ", end="")
            if isinstance(v, dict):
                print(f"subkeys={list(v.keys())[:8]}")
                for sk in list(v.keys())[:2]:
                    sv = v[sk]
                    print(f"    {sk}: type={type(sv)}", end="")
                    if hasattr(sv, 'shape'): print(f", shape={sv.shape}")
                    elif hasattr(sv, '__len__'): print(f", len={len(sv)}")
                    else: print(f", val={sv}")
            elif isinstance(v, list):
                print(f"len={len(v)}")
                if v: print(f"    [0]: type={type(v[0])}")
            elif hasattr(v, 'shape'):
                print(f"shape={v.shape}")
            else:
                print(repr(v)[:100])
    elif isinstance(data, list):
        print(f"  [0]: type={type(data[0])}")
        if isinstance(data[0], dict):
            pprint.pprint({k: type(v) for k,v in data[0].items()})
