"""DQN/DDQN 智能体在 AMR 路径规划环境中的训练循环。

用法：  python train.py --profile <name>     （读取 configs/<name>.json）
        python train.py --self-check         （仅验证 CUDA 和 import）

结构（1500+ 行）
-----------------------
辅助函数：
    moving_average()                   训练曲线平滑。
DQfD 专家支持：
    forest_demo_target()               预填充 demo 数量。
    forest_expert_action()             向 Hybrid A* 专家查询单步动作。
    collect_forest_demos()             批量填充经验回放池（专家演示）。

核心：
    train_one()                        训练一个 (env, algo) 组合 N 个 episode。
                                       包含 episode 循环、DQfD 预训练、
                                       周期性贪心评估和 checkpoint 保存。

CLI：
    build_parser()                     Argparse 定义（约 300 行参数）。
    main()                             入口：加载配置 -> 遍历 envs x algos -> train_one。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from ugv_dqn.reward_norm import RunningRewardNormalizer
from ugv_dqn.runtime import configure_runtime, select_device, torch_runtime_info
from ugv_dqn.runs import create_run_dir, resolve_experiment_dir

configure_runtime()

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from ugv_dqn.agents import AgentConfig, DQNFamilyAgent, parse_rl_algo
from ugv_dqn.config_io import apply_config_defaults, load_json, resolve_config_path, select_section
from ugv_dqn.env import UGVBicycleEnv
from ugv_dqn.forest_policy import forest_compute_next_mask, forest_select_action
from ugv_dqn.maps import FOREST_ENV_ORDER, REALMAP_ENV_ORDER, get_map_spec


# ===========================================================================
# 绘图辅助函数
# ===========================================================================

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = int(window)
    if x.size < w:
        return x
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same")


def plot_training_diagnostics(df_diag: pd.DataFrame, *, out_path: Path) -> None:
    """绘制 loss 曲线、epsilon 衰减和 Q 值分布。"""
    if df_diag.empty:
        return

    envs = [str(x) for x in df_diag["env"].drop_duplicates().tolist()]
    if not envs:
        return

    algo_label = {
        "mlp-dqn": "MLP-DQN", "mlp-ddqn": "MLP-DDQN", "mlp-pddqn": "MLP-PDDQN",
        "cnn-dqn": "CNN-DQN", "cnn-ddqn": "CNN-DDQN", "cnn-pddqn": "CNN-PDDQN",
    }
    present = [str(x) for x in df_diag["algo"].dropna().drop_duplicates().tolist()]
    pref = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn")
    ordered = [a for a in pref if a in present] + [a for a in present if a not in pref]
    algo_defs = [(a, algo_label.get(a, a.upper())) for a in ordered]

    metrics: list[tuple[str, str]] = [
        ("loss", "Total loss"),
        ("td_loss", "TD loss"),
        ("epsilon", "Epsilon"),
        ("q_spread", "Q spread (max-min)"),
        ("q_mean", "Q mean"),
        ("q_std", "Q std"),
    ]

    rows_n = len(envs)
    cols_n = len(metrics)
    fig, axes = plt.subplots(
        rows_n, cols_n,
        figsize=(3.8 * cols_n, 2.8 * rows_n),
        sharex=False, sharey=False,
    )
    axes_arr = np.atleast_2d(axes)

    for i, env_name in enumerate(envs):
        for j, (col, title) in enumerate(metrics):
            ax = axes_arr[i, j]
            for algo, label in algo_defs:
                sub = df_diag[(df_diag["env"] == env_name) & (df_diag["algo"] == algo)].copy()
                if sub.empty or col not in sub.columns:
                    continue
                sub = sub.sort_values("episode")
                x = sub["episode"].to_numpy()
                y = pd.to_numeric(sub[col], errors="coerce").astype(float).to_numpy()
                y = np.where(np.isfinite(y), y, np.nan)
                ax.plot(x, y, alpha=0.15, linewidth=0.5, color=ax._get_lines.get_next_color())
                win = max(1, len(y) // 15)
                y_smooth = pd.Series(y).rolling(window=win, min_periods=1, center=True).mean().to_numpy()
                ax.plot(x, y_smooth, label=label, linewidth=1.2, color=ax.lines[-1].get_color())
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(str(env_name), fontsize=8)
            if i == rows_n - 1:
                ax.set_xlabel("Episode", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), fontsize=8, frameon=False)
    fig.suptitle("Training diagnostics: loss / epsilon / Q-value", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ===========================================================================
# DQfD 专家演示支持
# ===========================================================================

def forest_demo_target(*, learning_starts: int, batch_size: int) -> int:
    # Forest 长期运行若回放池初始未被成功专家轨迹主导，容易陷入"停住直到卡住"的局部最优。
    # 经验上 forest_a 需要约 20k demo 转移（上限）才能稳定模仿 + TD 引导。
    target = max(int(learning_starts) * 40, int(batch_size))
    return int(min(int(target), 20_000))


def forest_expert_action(
    env: UGVBicycleEnv,
    *,
    forest_expert: str,
    horizon_steps: int,
    w_clearance: float,
) -> int:
    h = max(1, int(horizon_steps))
    expert = str(forest_expert).lower().strip()
    if expert == "auto":
        expert = "hybrid_astar"

    if expert == "hybrid_astar":
        # 更安全的 Hybrid A* 跟踪，用于演示/引导探索。
        # 短视距的激进跟踪器在较难的地图上（尤其是 forest_a）容易碰撞。
        return env.expert_action_hybrid_astar(
            lookahead_points=5,
            horizon_steps=max(15, h),
            w_target=0.2,
            w_heading=0.2,
            w_clearance=float(w_clearance),
            w_speed=0.0,
        )

    if expert in {"cost_to_go", "ctg"}:
        return env.expert_action_cost_to_go(horizon_steps=max(15, h), min_od_m=0.0)

    raise ValueError("forest_expert must be one of: auto, hybrid_astar, cost_to_go")


# ===========================================================================
# 核心训练循环
# ===========================================================================

def collect_forest_demos(
    env: UGVBicycleEnv,
    *,
    target: int,
    seed: int,
    forest_curriculum: bool,
    curriculum_band_m: float,
    forest_random_start_goal: bool,
    forest_rand_min_cost_m: float,
    forest_rand_max_cost_m: float | None,
    forest_rand_fixed_prob: float,
    forest_rand_tries: int,
    forest_expert: str,
    forest_demo_horizon: int,
    forest_demo_w_clearance: float,
    forest_adm_horizon: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expert = str(forest_expert).lower().strip()
    if expert == "auto":
        expert = "cost_to_go" if bool(forest_random_start_goal) else "hybrid_astar"

    obs_dim = int(env.observation_space.shape[0])
    n = max(0, int(target))
    obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
    next_obs_buf = np.zeros((n, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n,), dtype=np.int64)
    rew_buf = np.zeros((n,), dtype=np.float32)
    next_mask_buf = np.ones((n, int(env.action_space.n)), dtype=np.bool_)
    done_buf = np.zeros((n,), dtype=np.float32)
    trunc_buf = np.zeros((n,), dtype=np.float32)

    added = 0
    demo_ep = 0
    demo_prog = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
    # 仅保留成功（到达目标）episode 的演示。
    # 否则，失败的专家 rollout 会主导 DQfD loss + demo 保留回放池，
    # 使策略锁定在退化的"停住直到卡住"行为上。
    max_demo_eps = 2000
    while added < n and demo_ep < int(max_demo_eps):
        opts = None
        if forest_random_start_goal:
            opts = {
                "random_start_goal": True,
                "rand_min_cost_m": float(forest_rand_min_cost_m),
                "rand_max_cost_m": forest_rand_max_cost_m,
                "rand_fixed_prob": float(forest_rand_fixed_prob),
                "rand_tries": int(forest_rand_tries),
            }
        elif forest_curriculum:
            p = float(demo_prog[demo_ep % int(demo_prog.size)])
            opts = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
        obs, _ = env.reset(seed=int(seed) + 50_000 + int(demo_ep), options=opts)
        done = False
        truncated = False
        reached = False
        ep: list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray]] = []
        while not (done or truncated):
            a = forest_expert_action(
                env,
                forest_expert=str(expert),
                horizon_steps=int(forest_demo_horizon),
                w_clearance=float(forest_demo_w_clearance),
            )
            next_obs, reward, done, truncated, info = env.step(int(a))
            next_mask = forest_compute_next_mask(env, horizon_steps=int(forest_adm_horizon))
            reached = bool(reached or bool(info.get("reached", False)))
            ep.append((obs, int(a), float(reward), next_obs, bool(done), bool(truncated), next_mask))
            obs = next_obs

        if bool(reached):
            if int(added + len(ep)) > int(n):
                break
            for o, a, r, no, d, tr, nm in ep:
                obs_buf[added] = o
                next_obs_buf[added] = no
                act_buf[added] = int(a)
                rew_buf[added] = float(r)
                next_mask_buf[added] = nm
                done_buf[added] = 1.0 if bool(d) else 0.0
                trunc_buf[added] = 1.0 if bool(tr) else 0.0
                added += 1
                if added >= n:
                    break
        demo_ep += 1

    return (
        obs_buf[:added],
        act_buf[:added],
        rew_buf[:added],
        next_obs_buf[:added],
        next_mask_buf[:added],
        done_buf[:added],
        trunc_buf[:added],
    )


def train_one(
    env: gym.Env,
    algo: str,
    *,
    episodes: int,
    seed: int,
    out_dir: Path,
    agent_cfg: AgentConfig,
    train_freq: int,
    learning_starts: int,
    forest_curriculum: bool,
    curriculum_band_m: float,
    curriculum_ramp: float,
    forest_demo_prefill: bool,
    forest_demo_pretrain_steps: int,
    forest_demo_horizon: int,
    forest_demo_w_clearance: float,
    forest_demo_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    forest_expert: str,
    forest_expert_exploration: bool,
    forest_action_shield: bool,
    forest_adm_horizon: int,
    forest_topk: int,
    forest_expert_prob_start: float,
    forest_expert_prob_final: float,
    forest_expert_prob_decay: float,
    forest_random_start_goal: bool,
    forest_rand_min_cost_m: float,
    forest_rand_max_cost_m: float | None,
    forest_rand_fixed_prob: float,
    forest_rand_tries: int,
    eval_every: int,
    eval_runs: int,
    eval_score_time_weight: float,
    save_every: int,
    progress: bool,
    device: torch.device,
    reward_clip: float = 0.0,
    reward_norm: bool = True,
    cnn_drop_edt: bool = False,
) -> tuple[DQNFamilyAgent, np.ndarray, list[dict[str, float | int]], list[dict[str, float]]]:
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    agent = DQNFamilyAgent(algo, obs_dim, n_actions, config=agent_cfg, seed=seed, device=device, cnn_drop_edt=cnn_drop_edt)

    returns = np.zeros((episodes,), dtype=np.float32)
    global_step = 0
    eval_history: list[dict[str, float | int]] = []
    diag_history: list[dict[str, float]] = []
    rew_normalizer = RunningRewardNormalizer(clip=float(reward_clip)) if bool(reward_norm) else None

    best_score: tuple[int, int, int] = (-1, -10**18, 0)
    best_q: dict[str, torch.Tensor] | None = None
    best_q_target: dict[str, torch.Tensor] | None = None
    best_train_steps: int = 0
    pretrain_q: dict[str, torch.Tensor] | None = None
    pretrain_q_target: dict[str, torch.Tensor] | None = None
    pretrain_train_steps: int = 0
    # MinTD：追踪训练过程中 TD loss 最小的 checkpoint
    min_td_loss_val: float = float("inf")
    min_td_q: dict[str, torch.Tensor] | None = None
    min_td_q_target: dict[str, torch.Tensor] | None = None
    min_td_train_steps: int = 0
    explore_rng = np.random.default_rng(seed + 777)

    def episode_score(*, reached: bool, collision: bool, steps: int, ret: float) -> tuple[int, int, int]:
        """优先级：到达 > 存活（超时） > 碰撞（同级别下优先更高回报）。"""
        if bool(reached):
            # 优先更高回报，其次更少步数。
            return (2, int(1_000_000 * float(ret)), -int(steps))
        if bool(collision):
            return (0, int(1_000_000 * float(ret)), -int(steps))
        # 超时 / 未到达目标。
        return (1, int(1_000_000 * float(ret)), -int(steps))

    def clone_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    # 将 checkpoint 选择和指标记录与评估分布匹配：
    # - 固定起点训练 => 在规范起点/终点上评估
    # - 随机起点/终点训练 => 在一小批固定采样的 (start, goal) 对上评估
    eval_reset_options_list: list[dict[str, object] | None] = [None]
    if bool(forest_random_start_goal) and isinstance(env, UGVBicycleEnv):
        eval_reset_options_list = []
        n_eval = max(1, int(eval_runs))
        for i in range(n_eval):
            env.reset(
                seed=int(seed) + 90_000 + int(i),
                options={
                    "random_start_goal": True,
                    "rand_min_cost_m": float(forest_rand_min_cost_m),
                    "rand_max_cost_m": forest_rand_max_cost_m,
                    "rand_fixed_prob": float(forest_rand_fixed_prob),
                    "rand_tries": int(forest_rand_tries),
                },
            )
            eval_reset_options_list.append(
                {
                    "start_xy": (int(env.start_xy[0]), int(env.start_xy[1])),
                    "goal_xy": (int(env.goal_xy[0]), int(env.goal_xy[1])),
                }
            )

    def sync_cuda() -> None:
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def forest_expert_action_local() -> int:
        if not isinstance(env, UGVBicycleEnv):
            raise RuntimeError("forest_expert_action called for non-forest env")
        expert = str(forest_expert).lower().strip()
        if expert == "auto":
            expert = "cost_to_go" if bool(forest_random_start_goal) else "hybrid_astar"
        return forest_expert_action(
            env,
            forest_expert=str(expert),
            horizon_steps=int(forest_demo_horizon),
            w_clearance=float(forest_demo_w_clearance),
        )

    # 仅 Forest：用少量短专家 rollout 预填充回放池。
    # 这是 off-policy 数据（对 Q-learning 有效），显著降低了
    # 收敛到退化的"停住直到卡住"行为的概率。
    if forest_demo_prefill and isinstance(env, UGVBicycleEnv) and learning_starts > 0:
        demo_target = forest_demo_target(learning_starts=int(learning_starts), batch_size=int(agent_cfg.batch_size))
        if forest_demo_data is not None:
            obs_buf, act_buf, rew_buf, next_obs_buf, next_mask_buf, done_buf, trunc_buf = forest_demo_data
            n = int(min(int(demo_target), int(obs_buf.shape[0])))
            for i in range(n):
                r_i = float(rew_buf[i])
                if rew_normalizer is not None:
                    rew_normalizer.update(r_i)
                if reward_clip > 0.0:
                    r_i = float(np.clip(r_i, -reward_clip, reward_clip))
                agent.observe(
                    obs_buf[i],
                    int(act_buf[i]),
                    r_i,
                    next_obs_buf[i],
                    bool(done_buf[i] > 0.5),
                    demo=True,
                    truncated=bool(trunc_buf[i] > 0.5),
                    next_action_mask=next_mask_buf[i] if forest_action_shield else None,
                )
            global_step += int(n)
        else:
            # 收集比 learning_starts 更多的转移：静态全局地图上的模仿非常高效，
            # 保留多样化的 demo 集合可以提高鲁棒性——当学习的策略
            # 轻微偏离参考轨迹时仍能稳定（类 DAgger 效果）。
            demo_added = 0
            demo_ep = 0
            demo_prog = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
            max_demo_eps = 2000
            while demo_added < demo_target and demo_ep < int(max_demo_eps):
                opts = None
                if bool(forest_random_start_goal):
                    opts = {
                        "random_start_goal": True,
                        "rand_min_cost_m": float(forest_rand_min_cost_m),
                        "rand_max_cost_m": forest_rand_max_cost_m,
                        "rand_fixed_prob": float(forest_rand_fixed_prob),
                        "rand_tries": int(forest_rand_tries),
                    }
                elif forest_curriculum:
                    # 启用课程学习时，多样化演示起点以匹配训练起始状态分布。
                    p = float(demo_prog[demo_ep % int(demo_prog.size)])
                    opts = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
                obs, _ = env.reset(seed=seed + 50_000 + demo_ep, options=opts)
                done = False
                truncated = False
                reached = False
                ep: list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray]] = []
                while not (done or truncated):
                    a = forest_expert_action_local()
                    next_obs, reward, done, truncated, info = env.step(a)
                    next_mask = forest_compute_next_mask(env, horizon_steps=int(forest_adm_horizon))
                    reached = bool(reached or bool(info.get("reached", False)))
                    ep.append((obs, int(a), float(reward), next_obs, bool(done), bool(truncated), next_mask))
                    obs = next_obs
                if bool(reached):
                    for o, a, r, no, d, tr, nm in ep:
                        r_norm = float(r)
                        if rew_normalizer is not None:
                            rew_normalizer.update(r_norm)
                        if reward_clip > 0.0:
                            r_norm = float(np.clip(r_norm, -reward_clip, reward_clip))
                        agent.observe(
                            o,
                            int(a),
                            r_norm,
                            no,
                            bool(d),
                            demo=True,
                            truncated=bool(tr),
                            next_action_mask=nm if forest_action_shield else None,
                        )
                        demo_added += 1
                        global_step += 1
                    if demo_added >= demo_target:
                        break
                demo_ep += 1

        # 在 TD 学习之前对 demo 进行监督式热启动。
        pre_steps = int(max(0, int(forest_demo_pretrain_steps)))
        if pre_steps > 0:
            # 分块运行，一旦贪心（带掩码）策略能到达目标就提前停止。
            done_steps = 0
            chunk = 2000
            while done_steps < pre_steps:
                n = min(int(chunk), int(pre_steps - done_steps))
                agent.pretrain_on_demos(steps=int(n))
                done_steps += int(n)

                # 快速自检：使用与推理相同的可行动作掩码。
                obs_eval, _ = env.reset(seed=seed + 99_999)
                done_eval = False
                trunc_eval = False
                reached_eval = False
                while not (done_eval or trunc_eval):
                    a_eval = forest_select_action(
                        env, agent, obs_eval,
                        episode=0, explore=False,
                        horizon_steps=int(forest_adm_horizon),
                        topk=int(forest_topk),
                    )
                    obs_eval, _r, done_eval, trunc_eval, info_eval = env.step(int(a_eval))
                    if bool(info_eval.get("reached", False)):
                        reached_eval = True
                        break
                if reached_eval:
                    break

            # 模仿学习后同步目标网络，使后续 TD 更新从一致的网络对开始。
            agent.q_target.load_state_dict(agent.q.state_dict())

            # 保存模仿学习后策略的快照：在静态地图上它通常是最可靠的
            # 到达目标策略，而后续 TD 更新有时会发生漂移。
            pretrain_q = clone_state_dict(agent.q.state_dict())
            pretrain_q_target = clone_state_dict(agent.q_target.state_dict())
            pretrain_train_steps = int(agent._train_steps)

        # 回放池有有效转移后立即开始学习。
        global_step = max(int(global_step), int(learning_starts))

    def eval_action(obs_eval: np.ndarray) -> int:
        if isinstance(env, UGVBicycleEnv):
            return forest_select_action(
                env, agent, obs_eval,
                episode=0, explore=False,
                horizon_steps=int(forest_adm_horizon),
                topk=int(forest_topk),
            )
        return int(agent.act(obs_eval, episode=0, explore=False))

    def eval_greedy_metrics() -> dict[str, float]:
        successes = 0
        collisions = 0
        total_ret = 0.0
        total_steps = 0
        times_s: list[float] = []
        succ_path_lens_m: list[float] = []

        def extract_xy_m(info: dict[str, object]) -> tuple[float, float] | None:
            if "pose_m" in info:
                try:
                    x_m, y_m, _ = info["pose_m"]  # type: ignore[misc]
                    return (float(x_m), float(y_m))
                except Exception:
                    return None
            if "agent_xy" in info:
                try:
                    ax, ay = info["agent_xy"]  # type: ignore[misc]
                    if isinstance(env, UGVBicycleEnv):
                        return (float(ax) * float(env.cell_size_m), float(ay) * float(env.cell_size_m))
                except Exception:
                    return None
            return None

        for i, r_opts in enumerate(eval_reset_options_list):
            obs_eval, info0 = env.reset(seed=int(seed) + 99_999 + int(i), options=r_opts)
            done_eval = False
            trunc_eval = False
            steps_eval = 0
            ret_eval = 0.0
            t_eval = 0.0
            path_len_m = 0.0
            last_xy_m = extract_xy_m(dict(info0))
            last_info_eval: dict[str, object] = {}
            while not (done_eval or trunc_eval):
                steps_eval += 1
                sync_cuda()
                t0 = time.perf_counter()
                a_eval = eval_action(obs_eval)
                sync_cuda()
                t_eval += float(time.perf_counter() - t0)
                obs_eval, r, done_eval, trunc_eval, info_eval = env.step(int(a_eval))
                last_info_eval = dict(info_eval)
                ret_eval += float(r)
                xy_m = extract_xy_m(last_info_eval)
                if last_xy_m is not None and xy_m is not None:
                    path_len_m += float(math.hypot(float(xy_m[0]) - float(last_xy_m[0]), float(xy_m[1]) - float(last_xy_m[1])))
                last_xy_m = xy_m

            reached_eval = bool(last_info_eval.get("reached", False))
            collision_eval = bool(last_info_eval.get("collision", False) or last_info_eval.get("stuck", False))
            if reached_eval:
                successes += 1
                succ_path_lens_m.append(float(path_len_m))
            if collision_eval:
                collisions += 1
            total_ret += float(ret_eval)
            total_steps += int(steps_eval)
            times_s.append(float(t_eval))

        n = max(1, int(len(eval_reset_options_list)))
        sr = float(successes) / float(n)
        avg_path_length = float(np.mean(succ_path_lens_m)) if succ_path_lens_m else float("nan")
        inference_time_s = float(np.mean(times_s)) if times_s else 0.0

        w_t = float(eval_score_time_weight)
        base = float(avg_path_length) + float(w_t) * float(inference_time_s)
        denom = max(float(sr), 1e-6)
        planning_cost = float(base) / float(denom)
        if not (sr > 0.0 and math.isfinite(base)):
            planning_cost = float("inf")

        return {
            "success_rate": float(successes) / float(n),
            "collision_rate": float(collisions) / float(n),
            "avg_return": float(total_ret) / float(n),
            "avg_steps": float(total_steps) / float(n),
            "avg_path_length": float(avg_path_length),
            "inference_time_s": float(inference_time_s),
            "planning_cost": float(planning_cost),
        }

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            pbar = None
        else:
            pbar = tqdm(
                range(episodes),
                desc=f"Train {env.map_spec.name} {algo}",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )

    ep_iter = pbar if pbar is not None else range(episodes)
    for ep in ep_iter:
        reset_options = None
        if bool(forest_random_start_goal) and isinstance(env, UGVBicycleEnv):
            reset_options = {
                "random_start_goal": True,
                "rand_min_cost_m": float(forest_rand_min_cost_m),
                "rand_max_cost_m": forest_rand_max_cost_m,
                "rand_fixed_prob": float(forest_rand_fixed_prob),
                "rand_tries": int(forest_rand_tries),
            }
        elif forest_curriculum and isinstance(env, UGVBicycleEnv):
            p_raw = float(ep) / float(max(1, episodes - 1))
            ramp = max(1e-6, float(curriculum_ramp))
            p = float(np.clip(p_raw / ramp, 0.0, 1.0))
            reset_options = {"curriculum_progress": p, "curriculum_band_m": float(curriculum_band_m)}
        obs, _ = env.reset(seed=seed + ep, options=reset_options)
        adm_h = int(forest_adm_horizon)
        topk_k = int(forest_topk)
        ep_return = 0.0
        done = False
        truncated = False
        ep_steps = 0
        last_info: dict[str, object] = {}
        ep_buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, bool, np.ndarray | None]] = []
        pending_updates = 0

        while not (done or truncated):
            ep_steps += 1
            global_step += 1
            # Forest 稳定器：训练早期将专家混入行为策略。
            # Off-policy Q-learning 仍然有效，同时成功轨迹的频率
            # 足以支撑长期回报的 bootstrapping。
            used_expert = False
            if forest_expert_exploration and isinstance(env, UGVBicycleEnv):
                ramp = max(1e-6, float(forest_expert_prob_decay))
                t = float(np.clip((float(ep) / float(max(1, episodes - 1))) / ramp, 0.0, 1.0))
                p_exp = float(forest_expert_prob_start) + (float(forest_expert_prob_final) - float(forest_expert_prob_start)) * t
                p_exp = float(np.clip(p_exp, 0.0, 1.0))
                if explore_rng.random() < p_exp:
                    action = forest_expert_action_local()
                    used_expert = True
                else:
                    if forest_action_shield:
                        action = forest_select_action(
                            env, agent, obs,
                            episode=ep, explore=True,
                            horizon_steps=adm_h, topk=topk_k,
                            training_mode=True,
                        )
                    else:
                        action = agent.act(obs, episode=ep, explore=True)
            elif bool(forest_action_shield) and isinstance(env, UGVBicycleEnv):
                action = forest_select_action(
                    env, agent, obs,
                    episode=ep, explore=True,
                    horizon_steps=adm_h, topk=topk_k,
                    training_mode=True,
                )
            else:
                action = agent.act(obs, episode=ep, explore=True)
            next_obs, reward, done, truncated, info = env.step(action)
            last_info = dict(info)
            next_mask = None
            if bool(forest_action_shield) and isinstance(env, UGVBicycleEnv):
                next_mask = forest_compute_next_mask(env, horizon_steps=adm_h)
            # 时间限制截断不应视为终止状态用于 bootstrapping。
            # 仅当该 episode 到达目标时才将专家转移标记为 demo。
            # 失败的专家步骤仍是有效的 off-policy 数据，但不应被模仿/保留。
            # V9：在回放池中存储原始奖励；在 update() 采样时归一化。
            # 始终跟踪统计量以保持归一化器对采样的时效性。
            if rew_normalizer is not None:
                rew_normalizer.update(float(reward))
            if reward_clip > 0.0:
                reward = float(np.clip(float(reward), -reward_clip, reward_clip))
            ep_buffer.append(
                (
                    obs,
                    int(action),
                    float(reward),
                    next_obs,
                    bool(done),
                    bool(truncated),
                    bool(used_expert),
                    next_mask,
                )
            )
            ep_return += float(reward)

            if global_step >= learning_starts and (global_step % max(1, train_freq) == 0):
                pending_updates += 1

            obs = next_obs

        reached_ep = bool(last_info.get("reached", False))
        for o, a, r, no, d, tr, ue, nm in ep_buffer:
            agent.observe(
                o,
                int(a),
                float(r),
                no,
                bool(d),
                demo=bool(ue and reached_ep),
                truncated=bool(tr),
                next_action_mask=nm,
            )
        ep_losses: list[dict[str, float]] = []
        for _ in range(int(pending_updates)):
            loss_info = agent.update(rew_normalizer=rew_normalizer)
            if loss_info:
                ep_losses.append(loss_info)

        returns[ep] = float(ep_return)

        # --- 诊断信息：loss、epsilon、Q 值分布 ---
        ep_diag: dict[str, float] = {"episode": float(ep + 1), "epsilon": float(agent.epsilon(ep))}
        if ep_losses:
            for k in ("loss", "td_loss", "margin_loss", "ce_loss"):
                vals = [d[k] for d in ep_losses if k in d]
                ep_diag[k] = float(np.mean(vals)) if vals else 0.0
            # MinTD：记录 TD loss 最小时的 checkpoint（跳过 learning_starts 前的不稳定期）
            td_vals = [d["td_loss"] for d in ep_losses if "td_loss" in d]
            if td_vals and global_step >= int(learning_starts):
                mean_td = float(np.mean(td_vals))
                if mean_td < min_td_loss_val:
                    min_td_loss_val = mean_td
                    min_td_q = clone_state_dict(agent.q.state_dict())
                    min_td_q_target = clone_state_dict(agent.q_target.state_dict())
                    min_td_train_steps = int(agent._train_steps)
        # Q 值分布：对当前观测做前向传播（廉价，单样本）
        with torch.no_grad():
            obs_diag = agent._prep_obs(obs)
            q_vals = agent.q(torch.from_numpy(obs_diag).unsqueeze(0).to(agent.device)).squeeze(0)
            q_np = q_vals.cpu().numpy()
            ep_diag["q_mean"] = float(np.mean(q_np))
            ep_diag["q_std"] = float(np.std(q_np))
            ep_diag["q_max"] = float(np.max(q_np))
            ep_diag["q_min"] = float(np.min(q_np))
            ep_diag["q_spread"] = float(np.max(q_np) - np.min(q_np))
        if rew_normalizer is not None:
            ep_diag["rew_norm_mean"] = float(rew_normalizer._mean)
            ep_diag["rew_norm_std"] = float(rew_normalizer.std)
            ep_diag["rew_norm_count"] = int(rew_normalizer._count)
        diag_history.append(ep_diag)

        every = int(max(0, int(eval_every)))
        if every > 0 and ((ep + 1) % every == 0 or ep == 0 or ep == int(episodes - 1)):
            m = eval_greedy_metrics()
            eval_history.append(
                {
                    "episode": int(ep + 1),
                    "success_rate": float(m["success_rate"]),
                    "collision_rate": float(m["collision_rate"]),
                    "avg_return": float(m["avg_return"]),
                    "avg_steps": float(m["avg_steps"]),
                    "avg_path_length": float(m["avg_path_length"]),
                    "inference_time_s": float(m["inference_time_s"]),
                    "planning_cost": float(m["planning_cost"]),
                }
            )

        if pbar is not None:
            pbar.set_postfix(
                {
                    "ret": f"{ep_return:.3f}",
                    "eps": f"{agent.epsilon(ep):.3f}",
                    "steps": global_step,
                    "updates": int(agent._train_steps),
                },
                refresh=False,
            )

        reached = bool(last_info.get("reached", False))
        collision = bool(last_info.get("collision", False) or last_info.get("stuck", False))
        score = episode_score(reached=reached, collision=collision, steps=ep_steps, ret=ep_return)
        if score > best_score:
            best_score = score
            best_q = clone_state_dict(agent.q.state_dict())
            best_q_target = clone_state_dict(agent.q_target.state_dict())
            best_train_steps = int(agent._train_steps)

        # 周期性 checkpoint 保存。
        if save_every > 0 and (ep + 1) % save_every == 0:
            ckpt_dir = out_dir / "checkpoints" / env.map_spec.name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(ckpt_dir / f"{agent.algo}_ep{ep + 1:05d}.pt")

    final_q = clone_state_dict(agent.q.state_dict())
    final_q_target = clone_state_dict(agent.q_target.state_dict())
    final_train_steps = int(agent._train_steps)

    def eval_greedy(q_sd: dict[str, torch.Tensor], q_target_sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        agent.q.load_state_dict(q_sd)
        agent.q_target.load_state_dict(q_target_sd)

        # 规范（单次）评估。
        if not (bool(forest_random_start_goal) and isinstance(env, UGVBicycleEnv)):
            obs, _ = env.reset(seed=seed + 9999)
            done = False
            truncated = False
            steps = 0
            ret = 0.0
            last_info: dict[str, object] = {}
            while not (done or truncated):
                steps += 1
                if isinstance(env, UGVBicycleEnv):
                    a = forest_select_action(
                        env, agent, obs,
                        episode=0, explore=False,
                        horizon_steps=int(forest_adm_horizon),
                        topk=int(forest_topk),
                    )
                else:
                    a = agent.act(obs, episode=0, explore=False)
                obs, r, done, truncated, info = env.step(a)
                last_info = dict(info)
                ret += float(r)

            reached = bool(last_info.get("reached", False))
            collision = bool(last_info.get("collision", False) or last_info.get("stuck", False))
            return episode_score(reached=reached, collision=collision, steps=steps, ret=ret)

        # 随机起点/终点评估：使用一批固定采样的 (start, goal) 对。
        successes = 0
        total_ret = 0.0
        total_steps = 0
        for i, r_opts in enumerate(eval_reset_options_list):
            obs, _ = env.reset(seed=seed + 9999 + int(i), options=r_opts)
            done = False
            truncated = False
            steps = 0
            ret = 0.0
            last_info: dict[str, object] = {}
            while not (done or truncated):
                steps += 1
                a = forest_select_action(
                    env, agent, obs,
                    episode=0, explore=False,
                    horizon_steps=int(forest_adm_horizon),
                    topk=int(forest_topk),
                )
                obs, r, done, truncated, info = env.step(a)
                last_info = dict(info)
                ret += float(r)
            if bool(last_info.get("reached", False)):
                successes += 1
            total_ret += float(ret)
            total_steps += int(steps)

        n = max(1, int(len(eval_reset_options_list)))
        avg_ret = float(total_ret) / float(n)
        avg_steps = float(total_steps) / float(n)
        return (int(successes), int(1_000_000 * float(avg_ret)), -int(avg_steps))

    # 根据贪心评估性能在最终策略和最佳（探索）episode checkpoint 之间选择。
    best_greedy_score = eval_greedy(final_q, final_q_target)
    chosen_q, chosen_q_target, chosen_train_steps = final_q, final_q_target, final_train_steps

    if best_q is not None and best_q_target is not None:
        candidate_score = eval_greedy(best_q, best_q_target)
        if candidate_score > best_greedy_score:
            best_greedy_score = candidate_score
            chosen_q, chosen_q_target, chosen_train_steps = best_q, best_q_target, best_train_steps

    if pretrain_q is not None and pretrain_q_target is not None:
        candidate_score = eval_greedy(pretrain_q, pretrain_q_target)
        if candidate_score > best_greedy_score:
            best_greedy_score = candidate_score
            chosen_q, chosen_q_target, chosen_train_steps = pretrain_q, pretrain_q_target, pretrain_train_steps

    if min_td_q is not None and min_td_q_target is not None:
        candidate_score = eval_greedy(min_td_q, min_td_q_target)
        if candidate_score > best_greedy_score:
            best_greedy_score = candidate_score
            chosen_q, chosen_q_target, chosen_train_steps = min_td_q, min_td_q_target, min_td_train_steps

    agent.q.load_state_dict(chosen_q)
    agent.q_target.load_state_dict(chosen_q_target)
    agent._train_steps = int(chosen_train_steps)

    model_path = out_dir / "models" / env.map_spec.name / f"{agent.algo}.pt"
    agent.save(model_path)

    # 额外保存 MinTD checkpoint（TD loss 最小时的模型），供推理对比使用
    if min_td_q is not None and min_td_q_target is not None:
        agent.q.load_state_dict(min_td_q)
        agent.q_target.load_state_dict(min_td_q_target)
        agent._train_steps = int(min_td_train_steps)
        mintd_path = out_dir / "models" / env.map_spec.name / f"{agent.algo}_mintd.pt"
        agent.save(mintd_path)
        # 恢复最终选定的模型
        agent.q.load_state_dict(chosen_q)
        agent.q_target.load_state_dict(chosen_q_target)
        agent._train_steps = int(chosen_train_steps)

    return agent, returns, eval_history, diag_history


# ===========================================================================
# Argparse 与 CLI 入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train RL agents (default: DQN) and generate Fig. 13-style reward curves.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. Supports a combined file with {train:{...}, infer:{...}}. CLI flags override config.",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile name under configs/ (e.g. forest_a_3000 -> configs/forest_a_3000.json). Overrides configs/config.json.",
    )
    ap.add_argument(
        "--envs",
        nargs="*",
        default=list(FOREST_ENV_ORDER),
        help="Subset of envs: forest_a forest_b forest_c forest_d",
    )
    ap.add_argument(
        "--rl-algos",
        nargs="+",
        default=["mlp-dqn"],
        help=(
            "RL algorithms to train: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn (or 'all'). "
            "Legacy aliases: dqn ddqn iddqn cnn-iddqn. Default: mlp-dqn."
        ),
    )
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--edt-collision-margin", type=str, default="half",
                    choices=["half", "diag"],
                    help="EDT collision margin: 'half'=0.5*cell (default), 'diag'=sqrt(2)/2*cell.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--sensor-range", type=int, default=6)
    ap.add_argument(
        "--n-sectors",
        type=int,
        default=36,
        help="Forest lidar sectors (36=10°, 72=5°). Ignored for non-forest envs.",
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Experiment name/dir (bare names are stored under --runs-root).",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out is a bare name, store it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/train_<timestamp>/ to avoid mixing outputs.",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="Torch device selection (default: cuda).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
    )
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--ma-window", type=int, default=20, help="Moving average window for plotting (1=raw).")
    ap.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save a .pt checkpoint every N episodes (0 disables). Default: 10.",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run a greedy evaluation rollout every N episodes (0 disables; recommended for smoother learning curves).",
    )
    ap.add_argument(
        "--eval-runs",
        type=int,
        default=5,
        help=(
            "Number of evaluation rollouts per eval point. For --forest-random-start-goal this is the fixed (start,goal) "
            "batch size used throughout training."
        ),
    )
    ap.add_argument(
        "--eval-score-time-weight",
        type=float,
        default=0.5,
        help=(
            "Time weight (m/s) for the planning_cost metric written to training_eval.csv/.xlsx: "
            "planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)."
        ),
    )
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
    )
    ap.add_argument(
        "--forest-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only curriculum: start closer to the goal, then expand to the full start-goal distance.",
    )
    ap.add_argument(
        "--curriculum-band-m",
        type=float,
        default=2.0,
        help="Curriculum band width (meters) for sampling start states by cost-to-go.",
    )
    ap.add_argument(
        "--curriculum-ramp",
        type=float,
        default=0.35,
        help=(
            "Forest curriculum ramp fraction (0<r<=1). The fixed-start probability and curriculum distance "
            "reach 1.0 by r*episodes (smaller = harder sooner, but avoids train/test mismatch)."
        ),
    )
    ap.add_argument(
        "--forest-demo-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: prefill replay with expert rollouts (stabilizes training).",
    )
    ap.add_argument(
        "--forest-demo-pretrain-steps",
        type=int,
        default=50_000,
        help="Forest-only: supervised warm-start steps on demo transitions (behavior cloning + margin).",
    )
    ap.add_argument(
        "--forest-demo-horizon",
        type=int,
        default=15,
        help="Forest-only: expert horizon steps for demo prefill (constant action).",
    )
    ap.add_argument(
        "--forest-demo-w-clearance",
        type=float,
        default=0.8,
        help="Forest-only: expert clearance weight for demo prefill.",
    )
    ap.add_argument(
        "--forest-expert-exploration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forest-only: mix expert actions into the behavior policy (stabilizes long-horizon learning).",
    )
    ap.add_argument(
        "--forest-action-shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: apply an admissible-action mask (safety/progress shield) to the agent's actions during training. "
            "Recommended to keep enabled even when --no-forest-expert-exploration is used to avoid train/infer mismatch."
        ),
    )
    ap.add_argument(
        "--forest-adm-horizon",
        type=int,
        default=15,
        help="Forest-only: admissible-action horizon steps (unified with infer).",
    )
    ap.add_argument(
        "--forest-topk",
        type=int,
        default=10,
        help="Forest-only: try the top-k greedy actions before computing a full admissible-action mask.",
    )
    ap.add_argument(
        "--forest-expert",
        choices=("auto", "hybrid_astar", "cost_to_go"),
        default="auto",
        help="Forest-only: expert source used for demos / guided exploration.",
    )
    ap.add_argument(
        "--forest-random-start-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forest-only: randomize start/goal at each reset (goal-conditioned training).",
    )
    ap.add_argument(
        "--forest-rand-min-cost-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) when sampling random pairs.",
    )
    ap.add_argument(
        "--forest-rand-max-cost-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) when sampling random pairs (<=0 disables).",
    )
    ap.add_argument(
        "--forest-rand-fixed-prob",
        type=float,
        default=0.2,
        help="Forest-only: probability of using the canonical fixed start/goal instead of a random pair.",
    )
    ap.add_argument(
        "--forest-rand-tries",
        type=int,
        default=200,
        help="Forest-only: rejection-sampling tries per episode when sampling random start/goal pairs.",
    )
    ap.add_argument(
        "--forest-expert-prob-start",
        type=float,
        default=0.7,
        help="Forest-only: probability of using expert instead of the agent's epsilon-greedy action selection (start).",
    )
    ap.add_argument(
        "--forest-expert-prob-final",
        type=float,
        default=0.0,
        help="Forest-only: probability of using expert instead of the agent's epsilon-greedy action selection (final).",
    )
    ap.add_argument(
        "--forest-expert-prob-decay",
        type=float,
        default=0.6,
        help="Forest-only: decay fraction (0<d<=1) for expert exploration probability.",
    )
    ap.add_argument(
        "--eps-decay",
        type=int,
        default=None,
        help="Override AgentConfig.eps_decay. Default: auto = max(200, int(0.8 * episodes)).",
    )
    ap.add_argument(
        "--replay-capacity",
        type=int,
        default=None,
        help="Replay buffer capacity. Default: auto = max(100_000, episodes * 100).",
    )
    ap.add_argument("--gamma", type=float, default=None, help="Override AgentConfig.gamma (discount factor).")
    ap.add_argument("--learning-rate", type=float, default=None, help="Override AgentConfig.learning_rate.")
    ap.add_argument("--dueling", action="store_true", default=False, help="Enable Dueling DQN head (CNN only).")
    ap.add_argument("--mha", action="store_true", default=False, help="Enable Spatial MHA on CNN feature maps.")
    ap.add_argument("--mha-heads", type=int, default=4, help="Number of heads for Spatial MHA (default: 4).")
    ap.add_argument("--coord-attn", action="store_true", default=False, help="Enable Coordinate Attention (CVPR 2021).")
    ap.add_argument("--noisy", action="store_true", default=False, help="Enable NoisyNet linear layers (ICLR 2018).")
    ap.add_argument("--noisy-reset-interval", type=int, default=4, help="NoisyNet: resample noise every N update steps (default: 4).")
    ap.add_argument("--munchausen", action="store_true", default=False, help="Enable Munchausen DQN (Vieillard et al., NeurIPS 2020).")
    ap.add_argument("--m-alpha", type=float, default=0.9, help="Munchausen scaling coefficient (default: 0.9).")
    ap.add_argument("--m-tau", type=float, default=0.03, help="Munchausen entropy temperature (default: 0.03).")
    ap.add_argument("--m-lo", type=float, default=-1.0, help="Munchausen log-policy clamp lower bound (default: -1.0).")
    ap.add_argument("--fadc", action="store_true", default=False, help="Enable FADC conv layer (CVPR 2024).")
    ap.add_argument("--deform", action="store_true", default=False, help="Enable Deformable Conv v2 (torchvision).")
    ap.add_argument("--iqn", action="store_true", default=False, help="Enable IQN distributional head (ICML 2018).")
    ap.add_argument("--iqn-cos", type=int, default=64, help="IQN cosine embedding dimension (default: 64).")
    ap.add_argument("--iqn-quantiles", type=int, default=8, help="IQN number of quantile samples (default: 8).")
    ap.add_argument(
        "--target-update-tau",
        type=float,
        default=None,
        help="Override AgentConfig.target_update_tau for ALL algos (>0 forces Polyak soft updates).",
    )
    ap.add_argument(
        "--reward-clip",
        type=float,
        default=0.0,
        help="Clip per-step rewards to [-v, +v] before storing in replay (0 disables).",
    )
    ap.add_argument(
        "--reward-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable running-mean/std reward normalization (V8). Clip is used as safety net.",
    )
    ap.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.3,
        help="Goal position tolerance in meters (env.goal_tolerance_m). Default: 0.3.",
    )
    ap.add_argument(
        "--goal-speed-tol",
        type=float,
        default=999.0,
        help="Goal speed tolerance in m/s. 999=disabled. Try 0.5. Default: 999.0.",
    )
    ap.add_argument(
        "--reward-k-goal",
        type=float,
        default=0.0,
        help="Goal proximity shaping gain (V12). 0 disables; try 5.0. Default: 0.0.",
    )
    ap.add_argument(
        "--reward-k-eff",
        type=float,
        default=0.0,
        help="Efficiency penalty: penalise wasted motion (traveled but no progress). 0 disables. Default: 0.0.",
    )
    ap.add_argument("--reward-k-t", type=float, default=0.2, help="Per-step time penalty. Default: 0.2.")
    ap.add_argument("--reward-k-delta", type=float, default=1.5, help="Steering rate penalty. Default: 1.5.")
    ap.add_argument("--reward-k-kappa", type=float, default=0.2, help="Curvature penalty. Default: 0.2.")
    ap.add_argument("--reward-k-a", type=float, default=0.2, help="Acceleration smoothness penalty. Default: 0.2.")
    ap.add_argument("--reward-k-o", type=float, default=1.5, help="Near-obstacle penalty coefficient. Default: 1.5.")
    ap.add_argument("--reward-k-v", type=float, default=2.0, help="Narrow corridor speed coupling. Default: 2.0.")
    ap.add_argument("--reward-k-p", type=float, default=12.0, help="Progress reward gain. Default: 12.0.")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a training progress bar (default: on when running in a TTY).",
    )
    ap.add_argument(
        "--cnn-drop-edt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation: drop the EDT clearance channel from CNN input (keep only occ + cost). Default: False.",
    )
    ap.add_argument(
        "--scalar-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation: use only 11-dim scalar obs (no map channels). Default: False.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()

    pre_args, _ = ap.parse_known_args(argv)
    try:
        config_path = resolve_config_path(config=getattr(pre_args, "config", None), profile=getattr(pre_args, "profile", None))
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if config_path is not None:
        cfg_raw = load_json(Path(config_path))
        cfg = select_section(cfg_raw, section="train")
        apply_config_defaults(ap, cfg, strict=True)

    args = ap.parse_args(argv)
    forest_envs = set(FOREST_ENV_ORDER) | set(REALMAP_ENV_ORDER)
    if int(args.max_steps) == 300 and args.envs and all(str(e) in forest_envs for e in args.envs):
        args.max_steps = 600
    canonical_all = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn")
    raw_algos = [str(a).lower().strip() for a in (args.rl_algos or [])]
    if any(a == "all" for a in raw_algos):
        raw_algos = list(canonical_all)

    rl_algos: list[str] = []
    unknown = []
    for a in raw_algos:
        try:
            canonical, _arch, _base, _legacy = parse_rl_algo(a)
        except ValueError:
            unknown.append(a)
            continue
        if canonical not in rl_algos:
            rl_algos.append(canonical)

    if unknown:
        print(
            f"Unknown --rl-algos value(s): {', '.join(unknown)}. Choose from: "
            f"{' '.join(canonical_all)} (or 'all'). Legacy aliases: dqn ddqn iddqn cnn-iddqn.",
            file=sys.stderr,
        )
        return 2
    if not rl_algos:
        print(f"No RL algorithms selected (choose from: {' '.join(canonical_all)}).", file=sys.stderr)
        return 2
    args.rl_algos = rl_algos

    if args.self_check:
        info = torch_runtime_info()
        print(f"torch={info.torch_version}")
        print(f"cuda_available={info.cuda_available}")
        print(f"torch_cuda_version={info.torch_cuda_version}")
        print(f"cuda_device_count={info.device_count}")
        if info.device_names:
            print("cuda_devices=" + ", ".join(info.device_names))
        try:
            device = select_device(device=args.device, cuda_device=args.cuda_device)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"device_ok={device}")
        return 0

    try:
        device = select_device(device=args.device, cuda_device=args.cuda_device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    progress = bool(sys.stderr.isatty()) if args.progress is None else bool(args.progress)

    experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs, prefix="train")
    out_dir = run_paths.run_dir

    # 构建 AgentConfig，支持 CLI/JSON 可选覆盖。
    agent_kw: dict[str, object] = {}
    # eps_decay：显式指定 > 自动缩放（80% episodes，最小 200）
    if args.eps_decay is not None:
        agent_kw["eps_decay"] = int(args.eps_decay)
    else:
        agent_kw["eps_decay"] = max(200, int(0.8 * args.episodes))
    # replay_capacity：显式指定 > 自动缩放（episodes * 100，最小 100_000）
    if args.replay_capacity is not None:
        agent_kw["replay_capacity"] = int(args.replay_capacity)
    else:
        agent_kw["replay_capacity"] = max(100_000, args.episodes * 100)
    # eval_every：显式指定（>0）> 自动缩放（episodes // 30，最小 10，每次运行约 30 个评估点）
    _eval_every = args.eval_every if args.eval_every > 0 else max(10, args.episodes // 30)
    if args.gamma is not None:
        agent_kw["gamma"] = float(args.gamma)
    if args.learning_rate is not None:
        agent_kw["learning_rate"] = float(args.learning_rate)
    if args.dueling:
        agent_kw["dueling"] = True
    if args.mha:
        agent_kw["mha"] = True
    if args.mha_heads != 4:
        agent_kw["mha_heads"] = int(args.mha_heads)
    if args.coord_attn:
        agent_kw["coord_attn"] = True
    if args.noisy:
        agent_kw["noisy"] = True
    if args.noisy_reset_interval != 4:
        agent_kw["noisy_reset_interval"] = int(args.noisy_reset_interval)
    if args.munchausen:
        agent_kw["munchausen"] = True
    if args.m_alpha != 0.9:
        agent_kw["m_alpha"] = float(args.m_alpha)
    if args.m_tau != 0.03:
        agent_kw["m_tau"] = float(args.m_tau)
    if args.m_lo != -1.0:
        agent_kw["m_lo"] = float(args.m_lo)
    if args.fadc:
        agent_kw["fadc"] = True
    if args.deform:
        agent_kw["deform"] = True
    if args.iqn:
        agent_kw["iqn"] = True
    if args.iqn_cos != 64:
        agent_kw["iqn_cos"] = int(args.iqn_cos)
    if args.iqn_quantiles != 8:
        agent_kw["iqn_quantiles"] = int(args.iqn_quantiles)
    agent_cfg = AgentConfig(**agent_kw)  # type: ignore[arg-type]

    # 各算法配置。设置 --target-update-tau 时会覆盖所有算法。
    tau_override = float(args.target_update_tau) if args.target_update_tau is not None else None
    pddqn_tau = float(tau_override) if tau_override is not None else 0.01
    dqn_cfg = replace(agent_cfg, eps_start=0.6, n_step=3, **({"target_update_tau": tau_override} if tau_override is not None else {}))
    ddqn_cfg = replace(agent_cfg, eps_start=0.6, n_step=3, **({"target_update_tau": tau_override} if tau_override is not None else {}))
    pddqn_cfg = replace(agent_cfg, eps_start=0.6, n_step=3, target_update_tau=pddqn_tau)

    reward_clip = float(args.reward_clip)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    args_payload: dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_payload[k] = str(v)
        else:
            args_payload[k] = v

    (out_dir / "configs" / "run.json").write_text(
        json.dumps(
            {
                "kind": "train",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "configs" / "agent_config.json").write_text(json.dumps(asdict(agent_cfg), indent=2, sort_keys=True), encoding="utf-8")
    algo_cfgs = {
        "mlp-dqn": dqn_cfg,
        "mlp-ddqn": ddqn_cfg,
        "mlp-pddqn": pddqn_cfg,
        "cnn-dqn": dqn_cfg,
        "cnn-ddqn": ddqn_cfg,
        "cnn-pddqn": pddqn_cfg,
    }
    for algo in args.rl_algos:
        cfg = algo_cfgs.get(str(algo))
        if cfg is None:
            continue
        (out_dir / "configs" / f"agent_config_{algo}.json").write_text(
            json.dumps(asdict(cfg), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    all_rows: list[dict[str, float | int | str]] = []
    all_eval_rows: list[dict[str, float | int | str]] = []
    all_diag_rows: list[dict[str, float | int | str]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    algo_labels = {
        "mlp-dqn": "MLP-DQN",
        "mlp-ddqn": "MLP-DDQN",
        "mlp-pddqn": "MLP-PDDQN",
        "cnn-dqn": "CNN-DQN",
        "cnn-ddqn": "CNN-DDQN",
        "cnn-pddqn": "CNN-PDDQN",
    }

    for env_name in args.envs:
        spec = get_map_spec(env_name)
        env = UGVBicycleEnv(
            spec,
            max_steps=args.max_steps,
            cell_size_m=0.1,
            sensor_range_m=float(args.sensor_range),
            n_sectors=args.n_sectors,
            obs_map_size=int(args.obs_map_size),
            goal_tolerance_m=float(args.goal_tolerance),
            goal_speed_tol_m_s=float(args.goal_speed_tol),
            reward_k_goal=float(args.reward_k_goal),
            reward_k_eff=float(getattr(args, "reward_k_eff", 0.0)),
            reward_k_p=float(getattr(args, "reward_k_p", 12.0)),
            reward_k_t=float(getattr(args, "reward_k_t", 0.2)),
            reward_k_delta=float(getattr(args, "reward_k_delta", 1.5)),
            reward_k_kappa=float(getattr(args, "reward_k_kappa", 0.2)),
            reward_k_a=float(getattr(args, "reward_k_a", 0.2)),
            reward_k_o=float(getattr(args, "reward_k_o", 1.5)),
            reward_k_v=float(getattr(args, "reward_k_v", 2.0)),
            edt_collision_margin=getattr(args, "edt_collision_margin", "diag"),
            scalar_only=bool(getattr(args, "scalar_only", False)),
        )
        forest_demo_data = None
        if bool(args.forest_demo_prefill) and int(args.learning_starts) > 0:
            demo_target = forest_demo_target(learning_starts=int(args.learning_starts), batch_size=int(agent_cfg.batch_size))
            rand_max = None if float(args.forest_rand_max_cost_m) <= 0.0 else float(args.forest_rand_max_cost_m)
            forest_demo_data = collect_forest_demos(
                env,
                target=int(demo_target),
                seed=int(args.seed + 1000),
                forest_curriculum=bool(args.forest_curriculum),
                curriculum_band_m=float(args.curriculum_band_m),
                forest_random_start_goal=bool(args.forest_random_start_goal),
                forest_rand_min_cost_m=float(args.forest_rand_min_cost_m),
                forest_rand_max_cost_m=rand_max,
                forest_rand_fixed_prob=float(args.forest_rand_fixed_prob),
                forest_rand_tries=int(args.forest_rand_tries),
                forest_expert=str(args.forest_expert),
                forest_demo_horizon=int(args.forest_demo_horizon),
                forest_demo_w_clearance=float(args.forest_demo_w_clearance),
                forest_adm_horizon=int(args.forest_adm_horizon),
            )

        rand_max = None if float(args.forest_rand_max_cost_m) <= 0.0 else float(args.forest_rand_max_cost_m)

        env_curves: dict[str, np.ndarray] = {}
        env_eval_rows: dict[str, list[dict[str, float | int]]] = {}
        for algo in args.rl_algos:
            cfg = algo_cfgs[str(algo)]
            _, algo_returns, algo_eval, algo_diag = train_one(
                env,
                str(algo),
                episodes=args.episodes,
                # Forest 训练（全局地图 + 模仿热启动）对随机初始化敏感。
                # 在各算法间保持确定性 seed 偏移以保证公平比较。
                seed=args.seed + 1000,
                out_dir=out_dir,
                agent_cfg=cfg,
                train_freq=args.train_freq,
                learning_starts=args.learning_starts,
                forest_curriculum=bool(args.forest_curriculum),
                curriculum_band_m=float(args.curriculum_band_m),
                curriculum_ramp=float(args.curriculum_ramp),
                forest_demo_prefill=bool(args.forest_demo_prefill),
                forest_demo_pretrain_steps=int(args.forest_demo_pretrain_steps),
                forest_demo_horizon=int(args.forest_demo_horizon),
                forest_demo_w_clearance=float(args.forest_demo_w_clearance),
                forest_demo_data=forest_demo_data,
                forest_expert=str(args.forest_expert),
                forest_expert_exploration=bool(args.forest_expert_exploration),
                forest_action_shield=bool(args.forest_action_shield),
                forest_adm_horizon=int(args.forest_adm_horizon),
                forest_topk=int(args.forest_topk),
                forest_expert_prob_start=float(args.forest_expert_prob_start),
                forest_expert_prob_final=float(args.forest_expert_prob_final),
                forest_expert_prob_decay=float(args.forest_expert_prob_decay),
                forest_random_start_goal=bool(args.forest_random_start_goal),
                forest_rand_min_cost_m=float(args.forest_rand_min_cost_m),
                forest_rand_max_cost_m=rand_max,
                forest_rand_fixed_prob=float(args.forest_rand_fixed_prob),
                forest_rand_tries=int(args.forest_rand_tries),
                eval_every=_eval_every,
                eval_runs=int(args.eval_runs),
                eval_score_time_weight=float(args.eval_score_time_weight),
                save_every=int(args.save_every),
                progress=progress,
                device=device,
                reward_clip=float(reward_clip),
                reward_norm=bool(args.reward_norm),
                cnn_drop_edt=bool(getattr(args, "cnn_drop_edt", False)),
            )
            env_curves[str(algo)] = algo_returns
            env_eval_rows[str(algo)] = list(algo_eval)
            for row in algo_eval:
                all_eval_rows.append({"env": env_name, "algo": str(algo), **row})
            for row in algo_diag:
                all_diag_rows.append({"env": env_name, "algo": str(algo), **row})

        curves[env_name] = dict(env_curves)
        for ep in range(args.episodes):
            row: dict[str, float | int | str] = {"env": env_name, "episode": ep + 1}
            for algo, returns_arr in env_curves.items():
                row[f"{algo}_return"] = float(returns_arr[ep])
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "training_returns.csv", index=False)
    if all_eval_rows:
        df_eval = pd.DataFrame(all_eval_rows)
        df_eval.to_csv(out_dir / "training_eval.csv", index=False)
        try:
            df_eval.to_excel(out_dir / "training_eval.xlsx", index=False)
        except Exception as exc:
            print(f"Warning: failed to write training_eval.xlsx: {exc}", file=sys.stderr)
    if all_diag_rows:
        df_diag = pd.DataFrame(all_diag_rows)
        df_diag.to_csv(out_dir / "training_diagnostics.csv", index=False)
        try:
            plot_training_diagnostics(df_diag, out_path=out_dir / "training_diagnostics.png")
        except Exception as exc:
            print(f"Warning: failed to write training_diagnostics.png: {exc}", file=sys.stderr)

    print(f"Wrote: {out_dir / 'training_returns.csv'}")
    if all_eval_rows:
        print(f"Wrote: {out_dir / 'training_eval.csv'}")
        if (out_dir / "training_eval.xlsx").exists():
            print(f"Wrote: {out_dir / 'training_eval.xlsx'}")
    if all_diag_rows:
        print(f"Wrote: {out_dir / 'training_diagnostics.csv'}")
        if (out_dir / "training_diagnostics.png").exists():
            print(f"Wrote: {out_dir / 'training_diagnostics.png'}")
    print(f"Wrote models under: {out_dir / 'models'}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
