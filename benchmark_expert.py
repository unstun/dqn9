"""最小化 Expert CTG 单条轨迹计时脚本。

无经验池、无神经网络、无任何额外开销。
纯粹测量: env.reset() + 循环 expert_action_cost_to_go() + env.step() 的耗时。
"""
import time
import numpy as np
from ugv_dqn.env import UGVBicycleEnv
from ugv_dqn.maps import get_map_spec

# ── 参数 ──
SEED = 110
N_RUNS = 20
HORIZON = 15
MAX_STEPS = 1000
ENV_NAME = "forest_a"

spec = get_map_spec(ENV_NAME)
env = UGVBicycleEnv(spec, max_steps=MAX_STEPS)

times = []
steps_list = []
results = []

for i in range(N_RUNS):
    seed_i = SEED + i
    obs, info = env.reset(seed=seed_i)

    t0 = time.perf_counter()
    done, truncated, steps, reached = False, False, 0, False
    while not (done or truncated) and steps < MAX_STEPS:
        a = env.expert_action_cost_to_go(horizon_steps=HORIZON, min_od_m=0.0)
        obs, rew, done, truncated, info = env.step(a)
        steps += 1
        if info.get("reached"):
            reached = True
            break
    elapsed = time.perf_counter() - t0

    times.append(elapsed)
    steps_list.append(steps)
    results.append(reached)
    tag = "OK" if reached else "FAIL"
    print(f"  run {i+1:2d}  seed={seed_i}  {tag}  steps={steps:3d}  time={elapsed:.4f}s")

env.close()

# ── 汇总 ──
arr = np.array(times)
print(f"\n{'='*50}")
print(f"Expert CTG Benchmark  (H={HORIZON}, {N_RUNS} runs)")
print(f"{'='*50}")
print(f"成功率:  {sum(results)}/{N_RUNS} = {sum(results)/N_RUNS*100:.0f}%")
print(f"平均耗时: {arr.mean():.4f}s  ± {arr.std():.4f}s")
print(f"最快:     {arr.min():.4f}s")
print(f"最慢:     {arr.max():.4f}s")
print(f"平均步数: {np.mean(steps_list):.1f}")
