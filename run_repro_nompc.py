#!/usr/bin/env python3
"""Batch launcher: Table2 no-MPC rerun, 10 seeds x 2 suites in parallel."""

import json, os, subprocess, pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJ = pathlib.Path("/home/ubuntu/DQN9")
CONDA = "/home/ubuntu/miniconda3/bin/conda"
ENV = "ros2py310"
SEEDS = list(range(110, 120))

TEMPLATES = {
    "long": PROJ / "configs" / "repro_20260324_nompc_sr_long.json",
    "short": PROJ / "configs" / "repro_20260324_nompc_sr_short.json",
}

CFG_DIR = PROJ / "configs" / "repro_nompc"


def make_config(suite: str, seed: int) -> pathlib.Path:
    """Generate per-seed config from template."""
    with open(TEMPLATES[suite]) as f:
        cfg = json.load(f)
    cfg["infer"]["seed"] = seed
    cfg["infer"]["out"] = f"repro_20260324_nompc_sr_{suite}"
    out = CFG_DIR / f"nompc_s{seed}_{suite}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    return out


def run_one(suite: str, seed: int) -> str:
    cfg = make_config(suite, seed)
    cmd = [
        CONDA, "run", "--cwd", str(PROJ), "-n", ENV,
        "python", "infer.py", "--profile", str(cfg),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    tag = f"[{suite}/s{seed}]"
    if r.returncode != 0:
        return f"{tag} FAIL\n{r.stderr[-500:]}"
    return f"{tag} OK"


def main():
    tasks = [(s, seed) for s in ["long", "short"] for seed in SEEDS]
    print(f"Launching {len(tasks)} tasks (10 seeds x 2 suites)...")
    with ProcessPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(run_one, s, seed): (s, seed) for s, seed in tasks}
        for fut in as_completed(futs):
            print(fut.result())
    print("All done.")


if __name__ == "__main__":
    main()
