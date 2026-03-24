"""批量 seed 扫描: 32 seeds × 2 suites × 4 规划器, 每 seed 50 runs。

生成 64 个 config 文件, 然后并行执行。
4 规划器: MD-DDQN (RL), Expert CTG, RRT*, Hybrid A*
"""
import json
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── 参数 ──
SEEDS = list(range(100, 132))  # 32 seeds: 100-131
RUNS = 50
MODEL_DIR = "abl_diag10k_kt02_cnn_ddqn_md/train_20260315_120749"
CONDA = os.path.expanduser("~/miniconda3/bin/conda")
ENV_NAME = "ros2py310"
PROJ = os.path.expanduser("~/DQN9")
MAX_PARALLEL = 16  # 匹配 CPU 核心数

SUITES = {
    "long": {"rand_min_cost_m": 18.0, "rand_max_cost_m": 0.0},
    "short": {"rand_min_cost_m": 6.0, "rand_max_cost_m": 14.0},
}

BASE_CONFIG = {
    "envs": ["realmap_a"],
    "baselines": ["rrt_star", "lo_hybrid_astar"],
    "rl_algos": ["cnn-ddqn"],
    "expert_baseline": True,
    "rl_mpc_track": False,
    "forest_baseline_rollout": True,
    "forest_baseline_mpc_candidates": 256,
    "random_start_goal": True,
    "rand_two_suites": False,
    "runs": RUNS,
    "plot_pair_runs": False,
    "rand_tries": 600,
    "rand_reject_unreachable": True,
    "filter_all_succeed": False,
    "max_steps": 600,
    "goal_tolerance": 1.0,
    "goal_speed_tol": 999.0,
    "edt_collision_margin": "diag",
    "baseline_timeout": 10.0,
    "kpi_time_mode": "rollout",
    "composite_w_path_time": 1.0,
    "composite_w_avg_curvature": 0.6,
    "composite_w_planning_time": 0.2,
    "device": "cuda",
    "cuda_device": 0,
    "cnn_drop_edt": True,
    "models": MODEL_DIR,
}


def generate_configs():
    """生成 64 个 config 文件到 configs/sweep/ 目录"""
    cfg_dir = Path(PROJ) / "configs" / "sweep"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for seed in SEEDS:
        for suite, cost_range in SUITES.items():
            name = f"sweep_s{seed}_{suite}"
            cfg = dict(BASE_CONFIG)
            cfg["seed"] = seed
            cfg["out"] = f"sweep/{name}"
            cfg["rand_min_cost_m"] = cost_range["rand_min_cost_m"]
            cfg["rand_max_cost_m"] = cost_range["rand_max_cost_m"]

            cfg_path = cfg_dir / f"{name}.json"
            with open(cfg_path, "w") as f:
                json.dump({"infer": cfg}, f, indent=2)
            tasks.append((name, str(cfg_path)))

    return tasks


def run_one(args):
    """执行单个推理任务"""
    name, cfg_path = args
    profile = f"sweep/{name}"
    cmd = [
        CONDA, "run", "--cwd", PROJ, "-n", ENV_NAME,
        "python", "infer.py", "--profile", profile,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=PROJ,
        )
        ok = result.returncode == 0
        # 检查是否生成了 KPI 文件
        kpi_file = Path(PROJ) / "runs" / f"sweep/{name}" / "realmap_a" / "table2_kpis.csv"
        if not kpi_file.exists():
            # 有时输出目录结构不同，搜索一下
            pass
        status = "OK" if ok else "FAIL"
        err_tail = result.stderr[-200:] if result.stderr else ""
        return name, status, err_tail
    except subprocess.TimeoutExpired:
        return name, "TIMEOUT", ""
    except Exception as e:
        return name, "ERROR", str(e)


def main():
    print(f"=== Seed Sweep: {len(SEEDS)} seeds × {len(SUITES)} suites = {len(SEEDS)*len(SUITES)} tasks ===")
    print(f"每个任务: {RUNS} runs × 4 规划器 (MD-DDQN, Expert, RRT*, Hybrid A*)")
    print(f"并行度: {MAX_PARALLEL}")
    print()

    tasks = generate_configs()
    print(f"已生成 {len(tasks)} 个 config 文件")

    done_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            name, status, err = fut.result()
            done_count += 1
            if status != "OK":
                fail_count += 1
            print(f"  [{done_count:2d}/{len(tasks)}] {name}: {status}" +
                  (f"  {err[:100]}" if status != "OK" else ""))

    print(f"\n完成: {done_count - fail_count}/{done_count} 成功, {fail_count} 失败")


if __name__ == "__main__":
    main()
