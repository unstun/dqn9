"""已训练 DQN 智能体的推理、评估与可视化。

用法:  python infer.py --profile <name>     (读取 configs/<name>.json)
       python infer.py --self-check         (仅验证 CUDA 和 import)

结构 (2700+ 行)
-----------------------
轨迹辅助:
    _save_trace_json()                  写入每次运行的轨迹元数据伴随 JSON 文件。
    PathTrace / ControlTrace / RolloutResult   rollout 输出的数据类。

Rollout 引擎:
    rollout_agent()                     在环境中运行已训练智能体一个 episode。
    rollout_agent_plan_then_track()     Hybrid A* 规划后 MPC 跟踪模式。
    rollout_tracked_path_mpc()          纯 MPC 路径跟踪（用于经典基线算法）。

模型工具:
    infer_checkpoint_obs_dim()          从 .pt checkpoint 读取 obs_dim。
    forest_legacy_obs_transform()       处理旧版观测格式。

KPI 与后处理:
    mean_kpi()                          跨运行求 KPI 均值。
    smooth_path()                       Chaikin 平滑封装。

可视化:
    (已移除可视化辅助函数)
    write_paths_figure()                多面板路径对比图。
    write_controls_figure()             转向角 / 速度 / 曲率随时间变化图。

CLI:
    build_parser()                      Argparse 定义（约 300 行）。
    main()                              入口: 加载模型 -> 遍历环境 x 算法
                                        -> rollout -> KPI 表格 -> 绘图。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ugv_dqn.config_io import apply_config_defaults, load_json, resolve_config_path, select_section
from ugv_dqn.runtime import configure_runtime, select_device, torch_runtime_info
from ugv_dqn.runs import create_run_dir, resolve_experiment_dir, resolve_models_dir

configure_runtime()

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize as scipy_minimize

from ugv_dqn.agents import AgentConfig, DQNFamilyAgent, parse_rl_algo
from ugv_dqn.baselines.pathplan import (
    default_ackermann_params,
    PlannerResult,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_hybrid_astar,
    plan_lo_hybrid_astar,
    plan_rrt_star,
    point_footprint,
)
from ugv_dqn.env import UGVBicycleEnv, bilinear_sample_2d
from ugv_dqn.forest_policy import forest_select_action
from ugv_dqn.maps import FOREST_ENV_ORDER, REALMAP_ENV_ORDER, get_map_spec
from ugv_dqn.maps.forest import check_bicycle_reachable
from ugv_dqn.metrics import KPI, avg_abs_curvature, max_corner_degree, num_path_corners, path_length
from ugv_dqn.smoothing import chaikin_smooth


# ===========================================================================
# 轨迹辅助函数与数据类
# ===========================================================================

def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def _save_trace_json(
    traces_dir: Path,
    csv_name: str,
    *,
    algorithm: str,
    cell_size_m: float,
    env_base: str,
    env_case: str,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    run_idx: int,
) -> None:
    """在 CSV 旁边写入轨迹元数据 JSON 伴随文件。"""
    _start_m = (float(start_xy[0]) * float(cell_size_m), float(start_xy[1]) * float(cell_size_m))
    _goal_m = (float(goal_xy[0]) * float(cell_size_m), float(goal_xy[1]) * float(cell_size_m))
    json_name = csv_name.replace(".csv", ".json")
    (traces_dir / json_name).write_text(
        json.dumps({
            "algorithm": str(algorithm),
            "cell_size_m": float(cell_size_m),
            "env_base": str(env_base),
            "env_case": str(env_case),
            "goal_m": list(_goal_m),
            "goal_xy": list(goal_xy),
            "kind": "trace",
            "map_grid_npz": f"maps/{_safe_slug(env_base)}__grid_y0_bottom.npz",
            "map_meta_json": f"maps/{_safe_slug(env_base)}__meta.json",
            "run_idx": int(run_idx),
            "start_m": list(_start_m),
            "start_xy": list(start_xy),
            "trace_csv": f"traces/{csv_name}",
        }, indent=2, sort_keys=False),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class PathTrace:
    path_xy_cells: list[tuple[float, float]]
    success: bool


@dataclass(frozen=True)
class ControlTrace:
    t_s: np.ndarray
    v_m_s: np.ndarray
    delta_rad: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    path_xy_cells: list[tuple[float, float]]
    compute_time_s: float
    reached: bool
    steps: int
    path_time_s: float
    controls: ControlTrace | None = None
    planning_time_s: float = 0.0
    tracking_time_s: float = 0.0
    trace_rows: list[dict[str, object]] | None = None


def _env_dt_s(env: gym.Env) -> float:
    if isinstance(env, UGVBicycleEnv):
        return float(env.model.dt)
    return 1.0


# ===========================================================================
# Rollout 引擎（智能体、规划+跟踪、MPC）
# ===========================================================================

def rollout_agent(
    env: gym.Env,
    agent: DQNFamilyAgent,
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    forest_adm_horizon: int = 15,
    forest_topk: int = 10,
    forest_min_od_m: float = 0.0,
    forest_min_progress_m: float = 1e-4,
    collect_controls: bool = False,
    collect_trace: bool = False,
) -> RolloutResult:
    def sync_cuda() -> None:
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    time_mode = str(time_mode).lower().strip()
    if time_mode not in {"rollout", "policy"}:
        raise ValueError("time_mode must be one of: rollout, policy")

    inference_time_s = 0.0
    sync_cuda()
    # 在 env.reset() 之前开始计时，使 rollout 模式包含
    # Dijkstra 代价场和栅格地图构建时间（与基线算法公平对比）。
    t_rollout0 = time.perf_counter()

    obs, _info0 = env.reset(seed=seed, options=reset_options)
    if obs_transform is not None:
        obs = obs_transform(obs)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(_env_dt_s(env))

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if bool(collect_controls) and isinstance(env, UGVBicycleEnv):
        t_series = [0.0]
        v_series = [float(getattr(env, "_v_m_s", 0.0))]
        delta_series = [float(getattr(env, "_delta_rad", 0.0))]

    trace_rows: list[dict[str, object]] | None = None
    if bool(collect_trace) and isinstance(env, UGVBicycleEnv):
        trace_rows = [{
            "step": 0,
            "x_m": float(env._x_m),
            "y_m": float(env._y_m),
            "theta_rad": float(env._psi_rad),
            "v_m_s": float(env._v_m_s),
            "delta_rad": float(env._delta_rad),
            "action": -1,
            "delta_dot_rad_s": 0.0,
            "a_m_s2": 0.0,
            "od_m": float(getattr(env, "_last_od_m", 0.0)),
            "collision": False,
            "reached": False,
            "stuck": False,
            "reward": 0.0,
        }]

    done = False
    truncated = False
    steps = 0
    reached = False
    adm_h = max(1, int(forest_adm_horizon))
    topk_k = max(1, int(forest_topk))
    min_od = float(forest_min_od_m)
    min_prog = float(forest_min_progress_m)

    while not (done or truncated) and steps < max_steps:
        steps += 1
        if time_mode == "policy":
            sync_cuda()
            t0 = time.perf_counter()
        if isinstance(env, UGVBicycleEnv):
            a = forest_select_action(
                env, agent, obs,
                episode=0, explore=False,
                horizon_steps=adm_h, topk=topk_k,
                min_od_m=min_od, min_progress_m=min_prog,
            )
        else:
            a = agent.act(obs, episode=0, explore=False)
        if time_mode == "policy":
            sync_cuda()
            inference_time_s += float(time.perf_counter() - t0)
        obs, rew, done, truncated, info = env.step(a)
        if obs_transform is not None:
            obs = obs_transform(obs)
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if trace_rows is not None:
            _a_id = int(a)
            _dd = float(env.action_table[_a_id, 0])
            _aa = float(env.action_table[_a_id, 1])
            px, py, pth = info.get("pose_m", (env._x_m, env._y_m, env._psi_rad))
            trace_rows.append({
                "step": int(steps),
                "x_m": float(px),
                "y_m": float(py),
                "theta_rad": float(pth),
                "v_m_s": float(info.get("v_m_s", env._v_m_s)),
                "delta_rad": float(info.get("delta_rad", env._delta_rad)),
                "action": _a_id,
                "delta_dot_rad_s": _dd,
                "a_m_s2": _aa,
                "od_m": float(info.get("od_m", float("nan"))),
                "collision": bool(info.get("collision", False)),
                "reached": bool(info.get("reached", False)),
                "stuck": bool(info.get("stuck", False)),
                "reward": float(rew),
            })
        if t_series is not None and v_series is not None and delta_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", float(getattr(env, "_v_m_s", 0.0)))))
            delta_series.append(float(info.get("delta_rad", float(getattr(env, "_delta_rad", 0.0)))))
        if info.get("reached"):
            reached = True
            break

    if time_mode == "rollout":
        sync_cuda()
        inference_time_s = float(time.perf_counter() - t_rollout0)
    controls = None
    if t_series is not None and v_series is not None and delta_series is not None:
        controls = ControlTrace(
            t_s=np.asarray(t_series, dtype=np.float64),
            v_m_s=np.asarray(v_series, dtype=np.float64),
            delta_rad=np.asarray(delta_series, dtype=np.float64),
        )
    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=float(inference_time_s),
        reached=bool(reached),
        steps=int(steps),
        path_time_s=float(steps) * dt_s,
        controls=controls,
        trace_rows=trace_rows,
    )


def rollout_expert(
    env: UGVBicycleEnv,
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    horizon_steps: int = 15,
    collect_controls: bool = False,
    collect_trace: bool = False,
) -> RolloutResult:
    """基于 cost-to-go 贪心专家的 rollout（与 DQfD 演示生成使用相同的专家策略）。"""
    t0 = time.perf_counter()
    obs, _info0 = env.reset(seed=seed, options=reset_options)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(env.model.dt)
    h = max(1, int(horizon_steps))

    # 控制序列记录
    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if bool(collect_controls):
        t_series = [0.0]
        v_series = [float(env._v_m_s)]
        delta_series = [float(env._delta_rad)]

    # 逐步轨迹记录
    trace_rows: list[dict[str, object]] | None = None
    if bool(collect_trace):
        trace_rows = [{
            "step": 0,
            "x_m": float(env._x_m),
            "y_m": float(env._y_m),
            "theta_rad": float(env._psi_rad),
            "v_m_s": float(env._v_m_s),
            "delta_rad": float(env._delta_rad),
            "action": -1,
            "delta_dot_rad_s": 0.0,
            "a_m_s2": 0.0,
            "od_m": float(getattr(env, "_last_od_m", 0.0)),
            "collision": False,
            "reached": False,
            "stuck": False,
            "reward": 0.0,
        }]

    done = False
    truncated = False
    steps = 0
    reached = False

    while not (done or truncated) and steps < max_steps:
        steps += 1
        a = env.expert_action_cost_to_go(horizon_steps=h, min_od_m=0.0)
        obs, rew, done, truncated, info = env.step(a)
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if trace_rows is not None:
            _a_id = int(a)
            _dd = float(env.action_table[_a_id, 0])
            _aa = float(env.action_table[_a_id, 1])
            px, py, pth = info.get("pose_m", (env._x_m, env._y_m, env._psi_rad))
            trace_rows.append({
                "step": int(steps),
                "x_m": float(px),
                "y_m": float(py),
                "theta_rad": float(pth),
                "v_m_s": float(info.get("v_m_s", env._v_m_s)),
                "delta_rad": float(info.get("delta_rad", env._delta_rad)),
                "action": _a_id,
                "delta_dot_rad_s": _dd,
                "a_m_s2": _aa,
                "od_m": float(info.get("od_m", float("nan"))),
                "collision": bool(info.get("collision", False)),
                "reached": bool(info.get("reached", False)),
                "stuck": bool(info.get("stuck", False)),
                "reward": float(rew),
            })
        if t_series is not None and v_series is not None and delta_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", float(env._v_m_s))))
            delta_series.append(float(info.get("delta_rad", float(env._delta_rad))))
        if info.get("reached"):
            reached = True
            break

    compute_time_s = float(time.perf_counter() - t0)
    controls = None
    if t_series is not None and v_series is not None and delta_series is not None:
        controls = ControlTrace(
            t_s=np.asarray(t_series, dtype=np.float64),
            v_m_s=np.asarray(v_series, dtype=np.float64),
            delta_rad=np.asarray(delta_series, dtype=np.float64),
        )
    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=compute_time_s,
        reached=bool(reached),
        steps=int(steps),
        path_time_s=float(steps) * dt_s,
        controls=controls,
        trace_rows=trace_rows,
    )


def rollout_agent_plan_then_track(
    env: UGVBicycleEnv,
    agent: DQNFamilyAgent,
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    forest_adm_horizon: int = 15,
    forest_topk: int = 10,
    forest_min_od_m: float = 0.0,
    forest_min_progress_m: float = 1e-4,
    collect_controls: bool = False,
    mpc_candidates: int = 256,
) -> RolloutResult:
    """两阶段 RL 推理：DQN 规划全局路径，MPC 跟踪执行。

    阶段 1（规划）：``rollout_agent`` 生成航路点。
    阶段 2（跟踪）：``rollout_tracked_path_mpc`` 沿航路点执行跟踪。
    """
    # --- 阶段 1：DQN 规划 ---
    plan_roll = rollout_agent(
        env,
        agent,
        max_steps=max_steps,
        seed=seed,
        reset_options=reset_options,
        time_mode=time_mode,
        obs_transform=obs_transform,
        forest_adm_horizon=forest_adm_horizon,
        forest_topk=forest_topk,
        forest_min_od_m=forest_min_od_m,
        forest_min_progress_m=forest_min_progress_m,
        collect_controls=False,
    )
    plan_time = float(plan_roll.compute_time_s)
    dqn_path = list(plan_roll.path_xy_cells)

    if len(dqn_path) < 2:
        return RolloutResult(
            path_xy_cells=dqn_path,
            compute_time_s=plan_time,
            reached=False,
            steps=0,
            path_time_s=0.0,
            planning_time_s=plan_time,
            tracking_time_s=0.0,
        )

    # --- 阶段 2：MPC 跟踪 ---
    # 构建 reset 选项以恢复完全相同的起点/终点。
    sx, sy = int(env.start_xy[0]), int(env.start_xy[1])
    gx, gy = int(env.goal_xy[0]), int(env.goal_xy[1])
    track_opts: dict[str, object] = dict(reset_options) if reset_options else {}
    track_opts["start_xy"] = (sx, sy)
    track_opts["goal_xy"] = (gx, gy)

    track_roll = rollout_tracked_path_mpc(
        env,
        dqn_path,
        max_steps=max_steps,
        seed=seed + 50_000,
        reset_options=track_opts,
        time_mode=time_mode,
        collect_controls=collect_controls,
        n_candidates=mpc_candidates,
    )
    track_time = float(track_roll.compute_time_s)

    return RolloutResult(
        path_xy_cells=track_roll.path_xy_cells,
        compute_time_s=plan_time + track_time,
        reached=track_roll.reached,
        steps=track_roll.steps,
        path_time_s=track_roll.path_time_s,
        controls=track_roll.controls,
        planning_time_s=plan_time,
        tracking_time_s=track_time,
    )


def rollout_tracked_path_mpc(
    env: UGVBicycleEnv,
    ref_path_xy_cells: list[tuple[float, float]],
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    trace_path: Path | None = None,
    lookahead_points: int = 10,
    horizon_steps: int = 15,
    n_candidates: int = 512,
    w_track: float = 5.0,
    w_heading: float = 2.0,
    w_clearance: float = 3.0,
    w_delta_rate: float = 1.0,
    w_v_rate: float = 0.5,
    obstacle_safety_margin: float = 0.5,
    collect_controls: bool = False,
) -> RolloutResult:
    """传统优化式 MPC 路径跟踪器，用于基线算法路径（仅限 forest 环境）。

    控制变量为 (δ, v)（转向角、速度），使用 scipy SLSQP 求解有限时域
    约束优化问题，每步只执行第一个控制量（滚动时域）。
    """
    time_mode = str(time_mode).lower().strip()
    if time_mode not in {"rollout", "policy"}:
        raise ValueError("time_mode must be one of: rollout, policy")

    obs, _info0 = env.reset(seed=seed, options=reset_options)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(_env_dt_s(env))

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if bool(collect_controls):
        t_series = [0.0]
        v_series = [float(getattr(env, "_v_m_s", 0.0))]
        delta_series = [float(getattr(env, "_delta_rad", 0.0))]
    trace_rows: list[dict[str, object]] | None = [] if trace_path is not None else None
    if trace_rows is not None:
        d_goal0 = float(env._distance_to_goal_m())
        alpha0 = float(env._goal_relative_angle_rad())
        reached0 = (d_goal0 <= float(env.goal_tolerance_m)) and (abs(alpha0) <= float(env.goal_angle_tolerance_rad))
        trace_rows.append(
            {
                "step": 0,
                "x_m": float(env._x_m),
                "y_m": float(env._y_m),
                "theta_rad": float(env._psi_rad),
                "v_m_s": float(env._v_m_s),
                "delta_rad": float(env._delta_rad),
                "delta_dot_rad_s": 0.0,
                "a_m_s2": 0.0,
                "od_m": float(getattr(env, "_last_od_m", 0.0)),
                "collision": bool(getattr(env, "_last_collision", False)),
                "reached": bool(reached0),
                "stuck": False,
            }
        )
    if len(ref_path_xy_cells) < 2:
        controls = None
        if t_series is not None and v_series is not None and delta_series is not None:
            controls = ControlTrace(
                t_s=np.asarray(t_series, dtype=np.float64),
                v_m_s=np.asarray(v_series, dtype=np.float64),
                delta_rad=np.asarray(delta_series, dtype=np.float64),
            )
        return RolloutResult(
            path_xy_cells=path,
            compute_time_s=0.0,
            reached=False,
            steps=0,
            path_time_s=0.0,
            controls=controls,
        )

    # ---- 模型参数 ----
    ref_xy_m = np.asarray(ref_path_xy_cells, dtype=np.float64) * float(env.cell_size_m)
    h = max(1, int(horizon_steps))
    la = max(1, int(lookahead_points))
    dt = float(env.model.dt)
    wheelbase = float(env.model.wheelbase_m)
    v_max = float(env.model.v_max_m_s)
    delta_max = float(env.model.delta_max_rad)
    dd_max = float(env.model.delta_dot_max_rad_s)
    a_max = float(env.model.a_max_m_s2)
    cell_m = float(env.cell_size_m)

    # 双圆足迹参数 + diag 碰撞边距，与 env 碰撞检测对齐
    fp_x1 = float(env.footprint.x1_m)
    fp_x2 = float(env.footprint.x2_m)
    r_col = float(env.footprint.radius_m) + float(env._half_cell_m)

    # warm start 缓存：上一步解平移作为下一步初始猜测
    prev_sol: np.ndarray | None = None

    def _find_nearest_idx(progress_idx: int) -> int:
        """在参考路径上找到距离当前位置最近的索引（单调递增）。"""
        x_m, y_m = float(env._x_m), float(env._y_m)
        lo = max(0, int(progress_idx) - 25)
        hi = min(len(ref_path_xy_cells), int(progress_idx) + 250)
        if hi <= lo:
            lo, hi = 0, len(ref_path_xy_cells)
        best_i, best_d2 = lo, float("inf")
        for i in range(lo, hi):
            rx, ry = ref_xy_m[i, 0], ref_xy_m[i, 1]
            d2 = (rx - x_m) ** 2 + (ry - y_m) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return max(int(progress_idx), int(best_i))

    def _get_ref_window(progress_idx: int) -> np.ndarray:
        """提取从 progress_idx 开始的 h+1 个参考点（米），不足则外推。"""
        end_i = min(len(ref_xy_m), int(progress_idx) + h * la + 1)
        window = ref_xy_m[int(progress_idx) : end_i]
        if window.shape[0] < h + 1:
            # 用最后一段方向线性外推
            last = window[-1] if window.shape[0] > 0 else ref_xy_m[-1]
            if window.shape[0] >= 2:
                direction = window[-1] - window[-2]
            else:
                direction = np.array([0.0, 0.0])
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm * cell_m
            pad_n = h + 1 - window.shape[0]
            pad = last[None, :] + np.arange(1, pad_n + 1)[:, None] * direction[None, :]
            window = np.concatenate([window, pad], axis=0)
        # 在 h+1 个点上均匀采样（沿弧长）
        indices = np.linspace(0, window.shape[0] - 1, h + 1).astype(int)
        return window[indices]

    def choose_controls_mpc(progress_idx: int) -> tuple[float, float, int]:
        """传统 MPC 求解：min J(δ_0..δ_{H-1}, v_0..v_{H-1}) s.t. 自行车运动学。"""
        nonlocal prev_sol

        progress_idx = _find_nearest_idx(progress_idx)
        ref_window = _get_ref_window(progress_idx)  # (h+1, 2)

        # 当前状态
        x0 = float(env._x_m)
        y0 = float(env._y_m)
        psi0 = float(env._psi_rad)
        v0 = float(env._v_m_s)
        delta0 = float(env._delta_rad)

        # 接近目标时降速
        gx_m = float(env.goal_xy[0]) * cell_m
        gy_m = float(env.goal_xy[1]) * cell_m
        d_goal = math.hypot(x0 - gx_m, y0 - gy_m)
        decel_radius = 3.0 * float(env.goal_tolerance_m)
        v_cruise = v_max * min(1.0, d_goal / max(1e-6, decel_radius))
        v_cruise = max(0.1, v_cruise)

        # 决策变量：[δ_0, ..., δ_{H-1}, v_0, ..., v_{H-1}]，共 2H 维
        # bounds
        lb_delta = np.full(h, -delta_max)
        ub_delta = np.full(h, +delta_max)
        lb_v = np.full(h, 0.0)
        ub_v = np.full(h, v_cruise)
        bounds = list(zip(lb_delta, ub_delta)) + list(zip(lb_v, ub_v))

        # 初始猜测：warm start 或匀速直行
        if prev_sol is not None and prev_sol.shape[0] == 2 * h:
            x0_guess = np.empty(2 * h, dtype=np.float64)
            # 平移：丢弃第 0 步，末尾复制最后一步
            x0_guess[:h - 1] = prev_sol[1:h]
            x0_guess[h - 1] = prev_sol[h - 1]
            x0_guess[h:2 * h - 1] = prev_sol[h + 1:2 * h]
            x0_guess[2 * h - 1] = prev_sol[2 * h - 1]
        else:
            # 默认：转向角=当前值，速度=巡航速度
            x0_guess = np.concatenate([
                np.full(h, delta0),
                np.full(h, min(v_cruise, v0 + a_max * dt * h * 0.5)),
            ])
        # 裁剪到 bounds
        for i, (lo_b, hi_b) in enumerate(bounds):
            x0_guess[i] = np.clip(x0_guess[i], lo_b, hi_b)

        def cost_and_forward(u: np.ndarray) -> float:
            """代价函数：前向仿真 + 跟踪误差 + 控制平滑性 + 障碍物惩罚。"""
            delta_seq = u[:h]
            v_seq = u[h:]

            x_k, y_k, psi_k, v_k = x0, y0, psi0, v0
            cur_delta = delta0
            J = 0.0

            for k in range(h):
                # (δ, v) → (δ̇, a)，裁剪到执行器限幅
                dd = np.clip((delta_seq[k] - cur_delta) / dt, -dd_max, +dd_max)
                acc = np.clip((v_seq[k] - v_k) / dt, -a_max, +a_max)

                # 自行车模型积分一步
                v_next = np.clip(v_k + acc * dt, -v_max, v_max)
                delta_next = np.clip(cur_delta + dd * dt, -delta_max, +delta_max)
                x_next = x_k + v_next * math.cos(psi_k) * dt
                y_next = y_k + v_next * math.sin(psi_k) * dt
                psi_next = psi_k + (v_next / wheelbase) * math.tan(delta_next) * dt

                # 参考点跟踪误差
                ref_x, ref_y = ref_window[k + 1, 0], ref_window[k + 1, 1]
                dx_err = x_next - ref_x
                dy_err = y_next - ref_y
                J += float(w_track) * (dx_err * dx_err + dy_err * dy_err)

                # 航向误差：期望朝向下一参考点
                desired_heading = math.atan2(ref_y - y_k, ref_x - x_k)
                heading_err = psi_next - desired_heading
                heading_err = (heading_err + math.pi) % (2.0 * math.pi) - math.pi
                J += float(w_heading) * (heading_err * heading_err)

                # 控制平滑性：转向角变化率 + 速度变化率
                if k > 0:
                    J += float(w_delta_rate) * ((delta_seq[k] - delta_seq[k - 1]) ** 2)
                    J += float(w_v_rate) * ((v_seq[k] - v_seq[k - 1]) ** 2)
                else:
                    J += float(w_delta_rate) * ((delta_seq[0] - delta0) ** 2)
                    J += float(w_v_rate) * ((v_seq[0] - v0) ** 2)

                # 障碍物惩罚：双圆足迹 + diag 碰撞边距，与 env 碰撞检测一致
                cos_psi = math.cos(psi_next)
                sin_psi = math.sin(psi_next)
                c1x = x_next + cos_psi * fp_x1
                c1y = y_next + sin_psi * fp_x1
                c2x = x_next + cos_psi * fp_x2
                c2y = y_next + sin_psi * fp_x2
                d1 = bilinear_sample_2d(env._dist_m, x=c1x / cell_m, y=c1y / cell_m, default=0.0)
                d2 = bilinear_sample_2d(env._dist_m, x=c2x / cell_m, y=c2y / cell_m, default=0.0)
                clearance = min(d1, d2) - r_col
                if clearance < obstacle_safety_margin:
                    J += float(w_clearance) * ((obstacle_safety_margin - clearance) ** 2)

                x_k, y_k, psi_k, v_k = x_next, y_next, psi_next, v_next
                cur_delta = delta_next

            return J

        result = scipy_minimize(
            cost_and_forward,
            x0_guess,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-6, "disp": False},
        )

        prev_sol = result.x.copy()
        opt_delta = float(result.x[0])
        opt_v = float(result.x[h])

        # 转换为 (δ̇, a) 用于 step_continuous
        delta_dot = float(np.clip((opt_delta - delta0) / dt, -dd_max, +dd_max))
        accel = float(np.clip((opt_v - v0) / dt, -a_max, +a_max))

        return delta_dot, accel, int(progress_idx)

    inference_time_s = 0.0
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    reached = False
    progress_idx = 0
    while not (done or truncated) and steps < max_steps:
        steps += 1
        t0 = time.perf_counter() if time_mode == "policy" else None
        delta_dot, accel, progress_idx = choose_controls_mpc(progress_idx)
        if t0 is not None:
            inference_time_s += float(time.perf_counter() - t0)
        obs, _, done, truncated, info = env.step_continuous(delta_dot_rad_s=float(delta_dot), a_m_s2=float(accel))
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if t_series is not None and v_series is not None and delta_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", float(getattr(env, "_v_m_s", 0.0)))))
            delta_series.append(float(info.get("delta_rad", float(getattr(env, "_delta_rad", 0.0)))))
        if trace_rows is not None:
            px, py, pth = info.get("pose_m", (env._x_m, env._y_m, env._psi_rad))
            trace_rows.append(
                {
                    "step": int(steps),
                    "x_m": float(px),
                    "y_m": float(py),
                    "theta_rad": float(pth),
                    "v_m_s": float(info.get("v_m_s", env._v_m_s)),
                    "delta_rad": float(info.get("delta_rad", env._delta_rad)),
                    "delta_dot_rad_s": float(delta_dot),
                    "a_m_s2": float(accel),
                    "od_m": float(info.get("od_m", float("nan"))),
                    "collision": bool(info.get("collision", False)),
                    "reached": bool(info.get("reached", False)),
                    "stuck": bool(info.get("stuck", False)),
                }
            )
        if info.get("reached"):
            reached = True
            break

    if time_mode == "rollout":
        inference_time_s = float(time.perf_counter() - t_rollout0)

    if trace_path is not None and trace_rows is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False)

    controls = None
    if t_series is not None and v_series is not None and delta_series is not None:
        controls = ControlTrace(
            t_s=np.asarray(t_series, dtype=np.float64),
            v_m_s=np.asarray(v_series, dtype=np.float64),
            delta_rad=np.asarray(delta_series, dtype=np.float64),
        )
    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=float(inference_time_s),
        reached=bool(reached),
        steps=int(steps),
        path_time_s=float(steps) * dt_s,
        controls=controls,
    )


# ===========================================================================
# 模型工具与 KPI 后处理
# ===========================================================================

def infer_checkpoint_obs_dim(path: Path) -> int:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "q_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    obs_dim = payload.get("obs_dim")
    if isinstance(obs_dim, (int, float)) and int(obs_dim) > 0:
        return int(obs_dim)

    sd = payload["q_state_dict"]
    w = sd.get("net.0.weight")
    if w is None:
        w = sd.get("feature.0.weight")
    if w is None:
        raise ValueError(f"Could not infer observation dim from checkpoint: {path}")
    return int(w.shape[1])


def forest_legacy_obs_transform(obs: np.ndarray) -> np.ndarray:
    """将当前 forest 观测 (11+n_sectors) 映射为旧版 (7+n_sectors)。"""
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.size < 11:
        return x
    return np.concatenate([x[:7], x[11:]]).astype(np.float32, copy=False)


def mean_kpi(kpis: list[KPI]) -> KPI:
    if not kpis:
        nan = float("nan")
        return KPI(
            avg_path_length=nan,
            path_time_s=nan,
            avg_curvature_1_m=nan,
            planning_time_s=nan,
            tracking_time_s=nan,
            num_corners=nan,
            inference_time_s=nan,
            max_corner_deg=nan,
        )
    return KPI(
        avg_path_length=float(np.mean([k.avg_path_length for k in kpis])),
        path_time_s=float(np.mean([k.path_time_s for k in kpis])),
        avg_curvature_1_m=float(np.mean([k.avg_curvature_1_m for k in kpis])),
        planning_time_s=float(np.mean([k.planning_time_s for k in kpis])),
        tracking_time_s=float(np.mean([k.tracking_time_s for k in kpis])),
        num_corners=float(np.mean([k.num_corners for k in kpis])),
        inference_time_s=float(np.mean([k.inference_time_s for k in kpis])),
        max_corner_deg=float(np.mean([k.max_corner_deg for k in kpis])),
    )


def smooth_path(path: list[tuple[float, float]], *, iterations: int, enabled: bool = False) -> list[tuple[float, float]]:
    """Chaikin 平滑封装。默认禁用（enabled=False），直接返回原始路径。"""
    if not path:
        return []
    if not enabled:
        return list(path)
    pts = np.array(path, dtype=np.float32)
    sm = chaikin_smooth(pts, iterations=max(0, int(iterations)))
    return [(float(x), float(y)) for x, y in sm]


# ===========================================================================
# Argparse 与 CLI 入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run inference and generate Fig.12 + Table II-style KPIs.")
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
        "--models",
        type=Path,
        default=Path("outputs"),
        help="Model source: experiment name/dir, run dir, or models dir.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help=(
            "Output experiment name/dir. If this resolves to the same experiment as --models, "
            "results are stored under that training run directory."
        ),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out/--models is a bare name, store/read it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/<timestamp>/ (or <train_run>/infer/<timestamp>/) to avoid mixing outputs.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5, help="Averaging runs for stochastic methods.")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show an inference progress bar (default: on when running in a TTY).",
    )
    ap.add_argument(
        "--plot-run-idx",
        type=int,
        default=0,
        help=(
            "When --random-start-goal is enabled, plot this sample index in fig12_paths.png so all algorithms share "
            "the same (start,goal) pair (default: 0)."
        ),
    )
    ap.add_argument(
        "--plot-pair-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --rand-two-suites is enabled, write one 2-panel path figure per run index "
            "(short + long) so each image contains a short and a long random pair."
        ),
    )
    ap.add_argument(
        "--plot-pair-runs-max",
        type=int,
        default=10,
        help="Maximum number of per-run short+long figures to write when --plot-pair-runs is enabled (<=0 disables cap).",
    )
    ap.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        help="Optional baselines to evaluate: hybrid_astar rrt_star (or 'all'). Default: none.",
    )
    ap.add_argument(
        "--rl-algos",
        nargs="+",
        default=["mlp-dqn"],
        help=(
            "RL algorithms to evaluate: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn (or 'all'). "
            "Legacy aliases: dqn ddqn iddqn cnn-iddqn. Default: mlp-dqn."
        ),
    )
    ap.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip loading/running RL agents (useful for baseline-only evaluation).",
    )
    ap.add_argument(
        "--expert-baseline",
        action="store_true",
        help="Include cost-to-go greedy expert as a baseline (same expert used for DQfD demo generation).",
    )
    ap.add_argument(
        "--expert-horizon",
        type=int,
        default=15,
        help="Expert rollout horizon steps (default: 15, matching training demo generation).",
    )
    ap.add_argument("--baseline-timeout", type=float, default=5.0, help="Planner timeout (seconds).")
    ap.add_argument("--hybrid-max-nodes", type=int, default=200_000, help="Hybrid A* node budget.")
    ap.add_argument("--rrt-max-iter", type=int, default=5_000, help="RRT* iteration budget.")
    ap.add_argument("--ha-smooth", type=int, default=1,
                    help="Hybrid A* CG trajectory smoothing (Dolgov §3): 1=enable, 0=disable.")
    ap.add_argument("--ha-rs-heuristic-max-dist", type=float, default=15.0,
                    help="Hybrid A*: RS heuristic disabled when holonomic h > this (meters). 0=always on.")
    ap.add_argument("--ha-xy-resolution", type=float, default=0.0,
                    help="Hybrid A*: search grid resolution (meters). 0=use map cell_size.")
    ap.add_argument("--ha-step-length", type=float, default=0.3,
                    help="Hybrid A*: motion primitive step length (meters).")
    ap.add_argument("--loha-lo-iterations", type=int, default=0,
                    help="LO-HA* LOA outer-loop iterations (0=skip LOA, use default params).")
    ap.add_argument("--edt-collision-margin", type=str, default="half",
                    choices=["half", "diag"],
                    help="EDT collision margin: 'half'=0.5*cell (default), 'diag'=sqrt(2)/2*cell.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--sensor-range", type=int, default=6)
    ap.add_argument(
        "--n-sectors",
        type=int,
        default=36,
        help="Forest lidar sectors (kept for backwards compatibility; not used by the global-map observation).",
    )
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="Torch device selection (default: cuda).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--score-time-weight",
        type=float,
        default=0.5,
        help=(
            "Time weight (m/s) for the composite planning_cost metric: "
            "planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)."
        ),
    )
    ap.add_argument(
        "--composite-w-path-time",
        type=float,
        default=1.0,
        help="Weight for path_time_s in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--composite-w-avg-curvature",
        type=float,
        default=1.0,
        help="Weight for avg_curvature_1_m in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--composite-w-planning-time",
        type=float,
        default=1.0,
        help="Weight for planning_time_s in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--kpi-time-mode",
        choices=("rollout", "policy"),
        default="policy",
        help=(
            "How to measure inference_time_s for RL rollouts. "
            "'rollout' includes the full rollout wall-clock time (including env.step); "
            "'policy' measures only action-selection compute time (Q forward + admissibility checks)."
        ),
    )
    ap.add_argument(
        "--forest-adm-horizon",
        type=int,
        default=15,
        help="Forest-only: admissible-action horizon steps for safe/progress-gated rollouts.",
    )
    ap.add_argument(
        "--forest-topk",
        type=int,
        default=10,
        help="Forest-only: try the top-k greedy actions before computing a full admissible-action mask.",
    )
    ap.add_argument(
        "--forest-min-progress-m",
        type=float,
        default=1e-4,
        help="Forest-only: minimum cost-to-go progress required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-min-od-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum clearance (OD) required by admissible-action gating.",
    )
    ap.add_argument(
        "--rl-mpc-track",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when enabled, RL algorithms first generate a global path "
            "(DQN planning), then MPC tracks it (like baselines). "
            "Separates planning_time (DQN) and tracking_time (MPC). Default: disabled."
        ),
    )
    ap.add_argument(
        "--forest-baseline-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: when baselines are enabled, roll out a tracking controller on the planned "
            "baseline path to report executed-trajectory KPIs (default: enabled). "
            "Disable with --no-forest-baseline-rollout."
        ),
    )
    ap.add_argument(
        "--forest-baseline-mpc-candidates",
        type=int,
        default=256,
        help="Forest-only: continuous control samples per MPC step for baseline tracking.",
    )
    ap.add_argument(
        "--forest-baseline-save-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --forest-baseline-rollout is enabled, save per-run executed baseline trajectories "
            "(x,y,theta,v,delta,controls,OD) under <run_dir>/traces/ as CSV."
        ),
    )
    ap.add_argument(
        "--save-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save per-run trajectory traces (CSV + JSON metadata) and map exports (NPZ + JSON) "
            "under <run_dir>/traces/ and <run_dir>/maps/. Covers all algorithms (RL + baselines)."
        ),
    )
    ap.add_argument(
        "--random-start-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forest-only: evaluate on random start/goal pairs (uses --runs samples per environment).",
    )
    ap.add_argument(
        "--rand-min-cost-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) when sampling random pairs.",
    )
    ap.add_argument(
        "--rand-max-cost-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) when sampling random pairs (<=0 disables).",
    )
    ap.add_argument(
        "--rand-fixed-prob",
        type=float,
        default=0.0,
        help="Forest-only: probability of using the canonical fixed start/goal instead of a random pair.",
    )
    ap.add_argument(
        "--rand-tries",
        type=int,
        default=200,
        help="Forest-only: rejection-sampling tries per sample when sampling random start/goal pairs.",
    )
    ap.add_argument(
        "--rand-reject-unreachable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: when --random-start-goal is enabled, resample until Hybrid A* succeeds "
            "(avoids unreachable start/goal pairs in narrow forests and keeps comparisons meaningful)."
        ),
    )
    ap.add_argument(
        "--rand-reject-max-attempts",
        type=int,
        default=5000,
        help="Forest-only: maximum sampling attempts to find reachable random (start,goal) pairs.",
    )
    ap.add_argument(
        "--filter-all-succeed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Post-filter: after all algorithms finish, keep only (env, run_idx) pairs where EVERY algorithm "
            "reached the goal.  Outputs additional *_filtered.{csv,md} tables.  Useful for fair path-quality "
            "comparison that removes the effect of success-rate differences."
        ),
    )
    ap.add_argument(
        "--filter-target-count",
        type=int,
        default=0,
        help=(
            "When --filter-all-succeed is enabled, keep at most this many all-succeed pairs "
            "(0 = keep all).  The first N pairs (by run_idx order) are retained."
        ),
    )
    ap.add_argument(
        "--load-pairs",
        type=Path,
        default=None,
        help=(
            "Load pre-saved (start, goal) pairs from a JSON file instead of random sampling. "
            "Overrides --runs with the number of loaded pairs.  Use pairs previously saved by "
            "--filter-all-succeed runs (allsuc_pairs.json) to avoid re-running the full batch."
        ),
    )
    ap.add_argument(
        "--rand-two-suites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --random-start-goal is enabled, evaluate two random-pair suites (short + long) in one run. "
            "This adds '/short' and '/long' rows to KPI tables and plots."
        ),
    )
    ap.add_argument(
        "--rand-short-min-cost-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) for the 'short' random-pair suite.",
    )
    ap.add_argument(
        "--rand-short-max-cost-m",
        type=float,
        default=14.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) for the 'short' random-pair suite (<=0 disables).",
    )
    ap.add_argument(
        "--rand-long-min-cost-m",
        type=float,
        default=18.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) for the 'long' random-pair suite.",
    )
    ap.add_argument(
        "--rand-long-max-cost-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) for the 'long' random-pair suite (<=0 disables).",
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
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
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
        raise SystemExit(str(exc))
    if config_path is not None:
        cfg_raw = load_json(Path(config_path))
        cfg = select_section(cfg_raw, section="infer")
        if "forest_baseline_controller" in cfg:
            print(
                "Warning: config key 'forest_baseline_controller' is deprecated and ignored; "
                "baseline rollouts always use MPC now.",
                file=sys.stderr,
            )
        apply_config_defaults(ap, cfg, strict=True, allow_unknown_prefixes=("_", "forest_baseline_controller"))

    args = ap.parse_args(argv)
    if int(getattr(args, "plot_run_idx", 0)) < 0:
        raise SystemExit("--plot-run-idx must be >= 0")
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
        raise SystemExit(
            f"Unknown --rl-algos value(s): {', '.join(unknown)}. Choose from: "
            f"{' '.join(canonical_all)} (or 'all'). Legacy aliases: dqn ddqn iddqn cnn-iddqn."
        )
    if not rl_algos and not getattr(args, "baselines", []):
        raise SystemExit(f"No RL algorithms selected (choose from: {' '.join(canonical_all)}).")
    args.rl_algos = rl_algos
    # 无 RL 算法时自动跳过模型解析
    if not rl_algos:
        args.skip_rl = True

    baseline_aliases = {
        "hybrid_astar": "hybrid_astar",
        "hybrid-a-star": "hybrid_astar",
        "hybrid": "hybrid_astar",
        "ha": "hybrid_astar",
        "rrt_star": "rrt_star",
        "rrt*": "rrt_star",
        "rrt": "rrt_star",
        "ss-rrt*": "rrt_star",
        "ss_rrt_star": "rrt_star",
        "ss_rrt*": "rrt_star",
        "lo_hybrid_astar": "lo_hybrid_astar",
        "lo-ha*": "lo_hybrid_astar",
        "lo_ha*": "lo_hybrid_astar",
        "loha": "lo_hybrid_astar",
        "lo-hybrid-a-star": "lo_hybrid_astar",
        "all": "all",
    }
    baselines: list[str] = []
    for raw in args.baselines:
        key = str(raw).strip().lower()
        if not key:
            continue
        mapped = baseline_aliases.get(key)
        if mapped is None:
            raise SystemExit(
                f"Unknown baseline {raw!r}. Options: hybrid_astar, rrt_star, lo_hybrid_astar, all (aliases: hybrid, rrt, rrt*, lo-ha*, loha)."
            )
        if mapped == "all":
            baselines = ["hybrid_astar", "rrt_star", "lo_hybrid_astar"]
            break
        if mapped not in baselines:
            baselines.append(mapped)

    if bool(args.skip_rl) and not baselines and not bool(getattr(args, "expert_baseline", False)):
        raise SystemExit("--skip-rl requires at least one baseline via --baselines or --expert-baseline.")

    if bool(getattr(args, "rand_two_suites", False)):
        if not bool(getattr(args, "random_start_goal", False)):
            raise SystemExit("--rand-two-suites requires --random-start-goal.")
        if not args.envs or any(str(e) not in forest_envs for e in args.envs):
            raise SystemExit("--rand-two-suites is forest-only (use e.g. --envs forest_a ...).")
        if int(getattr(args, "runs", 0)) <= 0:
            raise SystemExit("--rand-two-suites requires --runs >= 1.")
        short_min = float(getattr(args, "rand_short_min_cost_m", 0.0))
        short_max = float(getattr(args, "rand_short_max_cost_m", 0.0))
        long_min = float(getattr(args, "rand_long_min_cost_m", 0.0))
        long_max = float(getattr(args, "rand_long_max_cost_m", 0.0))
        if short_max > 0.0 and short_min > short_max:
            raise SystemExit("--rand-short-min-cost-m must be <= --rand-short-max-cost-m (or disable max via <=0).")
        if long_max > 0.0 and long_min > long_max:
            raise SystemExit("--rand-long-min-cost-m must be <= --rand-long-max-cost-m (or disable max via <=0).")

    if bool(getattr(args, "rand_two_suites", False)):
        expanded_envs: list[str] = []
        for e in (args.envs or []):
            base = str(e).split("::", 1)[0].strip()
            if not base:
                continue
            expanded_envs.append(f"{base}::short")
            expanded_envs.append(f"{base}::long")
        args.envs = expanded_envs

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

    progress = bool(sys.stderr.isatty()) if getattr(args, "progress", None) is None else bool(getattr(args, "progress"))
    tqdm = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
        except Exception:
            tqdm = None
        else:
            tqdm = _tqdm

    requested_experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    models_dir: Path | None = None
    if not bool(args.skip_rl):
        models_dir = resolve_models_dir(args.models, runs_root=args.runs_root)

        models_run_dir = models_dir.parent
        models_experiment_dir = models_run_dir.parent

        # 如果输出指向相同的实验（带时间戳的 runs）或相同的 run 目录（无时间戳的 runs），
        # 将推理输出附加到训练 run 下。
        requested_resolved = requested_experiment_dir.resolve(strict=False)
        models_run_resolved = models_run_dir.resolve(strict=False)
        models_experiment_resolved = models_experiment_dir.resolve(strict=False)

        if requested_resolved == models_run_resolved or requested_resolved == models_experiment_resolved:
            # 将推理输出附加到训练 run 下，避免创建同级时间戳 run。
            experiment_dir = models_run_dir / "infer"
        else:
            experiment_dir = requested_experiment_dir
    else:
        experiment_dir = requested_experiment_dir

    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs)
    out_dir = run_paths.run_dir

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
                "kind": "infer",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "models_dir": (str(models_dir) if models_dir is not None else None),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    rows_runs: list[dict[str, object]] = []
    paths_for_plot: dict[tuple[str, int], dict[str, PathTrace]] = {}
    controls_for_plot: dict[tuple[str, int], dict[str, ControlTrace]] = {}
    plot_meta: dict[tuple[str, int], dict[str, float]] = {}

    for env_name in args.envs:
        env_case = str(env_name)
        suite_tag: str | None = None
        env_base = str(env_case)
        if "::" in env_case:
            env_base, suite_tag_raw = env_case.split("::", 1)
            env_base = str(env_base).strip()
            suite_tag = str(suite_tag_raw).strip() or None

        env_label = f"Env. ({env_base})" if suite_tag is None else f"Env. ({env_base})/{suite_tag}"

        rand_min_cost_m = float(getattr(args, "rand_min_cost_m", 0.0))
        rand_max_cost_m = float(getattr(args, "rand_max_cost_m", 0.0))
        if suite_tag == "short":
            rand_min_cost_m = float(getattr(args, "rand_short_min_cost_m", rand_min_cost_m))
            rand_max_cost_m = float(getattr(args, "rand_short_max_cost_m", rand_max_cost_m))
        elif suite_tag == "long":
            rand_min_cost_m = float(getattr(args, "rand_long_min_cost_m", rand_min_cost_m))
            rand_max_cost_m = float(getattr(args, "rand_long_max_cost_m", rand_max_cost_m))

        spec = get_map_spec(env_base)
        env = UGVBicycleEnv(
            spec,
            max_steps=args.max_steps,
            cell_size_m=0.1,
            sensor_range_m=float(args.sensor_range),
            n_sectors=args.n_sectors,
            obs_map_size=int(args.obs_map_size),
            goal_tolerance_m=float(args.goal_tolerance),
            goal_speed_tol_m_s=float(args.goal_speed_tol),
            edt_collision_margin=getattr(args, "edt_collision_margin", "diag"),
            scalar_only=bool(getattr(args, "scalar_only", False)),
        )
        cell_size_m = 0.1
        grid = spec.obstacle_grid()

        env_paths_by_run: dict[int, dict[str, PathTrace]] = {}
        base_meta: dict[str, float] = {"cell_size_m": float(cell_size_m)}
        base_meta["goal_tol_cells"] = float(env.goal_tolerance_m) / float(cell_size_m)
        fp = forest_two_circle_footprint()
        base_meta["veh_length_cells"] = float(fp.length) / float(cell_size_m)
        base_meta["veh_width_cells"] = float(fp.width) / float(cell_size_m)

        # --save-traces：每个 env_base 导出一次地图
        _save_traces = bool(getattr(args, "save_traces", False))
        if _save_traces and isinstance(env, UGVBicycleEnv):
            _maps_dir = out_dir / "maps"
            _maps_dir.mkdir(parents=True, exist_ok=True)
            _grid_path = _maps_dir / f"{_safe_slug(env_base)}__grid_y0_bottom.npz"
            if not _grid_path.exists():
                np.savez_compressed(str(_grid_path), obstacle_grid=grid, cell_size_m=float(cell_size_m))
            _meta_path = _maps_dir / f"{_safe_slug(env_base)}__meta.json"
            if not _meta_path.exists():
                _fp = forest_two_circle_footprint()
                _map_meta = {
                    "env_base": str(env_base),
                    "cell_size_m": float(cell_size_m),
                    "grid_shape": list(grid.shape),
                    "grid_values": {"0": "free", "1": "obstacle"},
                    "canonical_start_xy": [int(spec.start_xy[0]), int(spec.start_xy[1])],
                    "canonical_goal_xy": [int(spec.goal_xy[0]), int(spec.goal_xy[1])],
                    "collision_footprint": {
                        "kind": "two_circle",
                        "radius_m": float(_fp.radius),
                        "x1_m": float(_fp.center_shift - _fp.center_offset),
                        "x2_m": float(_fp.center_shift + _fp.center_offset),
                    },
                    "goal": {
                        "position_tolerance_m": float(env.goal_tolerance_m),
                        "position_tolerance_cells": float(env.goal_tolerance_m) / float(cell_size_m),
                        "angle_tolerance_rad": float(env.goal_angle_tolerance_rad),
                    },
                    "convention": {
                        "grid_y0_bottom": True,
                        "origin": "bottom_left",
                        "x_axis": "+right",
                        "y_axis": "+up",
                        "trace_units": {"x_m_y_m": "meters"},
                    },
                }
                _meta_path.write_text(json.dumps(_map_meta, indent=2, sort_keys=False), encoding="utf-8")

        plot_run_idx = int(getattr(args, "plot_run_idx", 0))
        plot_run_indices: list[int] = [int(plot_run_idx)]
        multi_pair_plot = (
            bool(getattr(args, "random_start_goal", False))
            and isinstance(env, UGVBicycleEnv)
            and int(args.runs) >= 4
            and int(len(args.envs)) == 1
        )
        if multi_pair_plot:
            plot_run_indices = [(int(plot_run_idx) + k) % int(args.runs) for k in range(4)]

        # 存储全部 run 的路径轨迹，用于离线绘图和 CSV 导出。
        # control_run_indices 仍只存绘图面板所需的子集以节省内存。
        path_run_indices: set[int] = set(range(int(args.runs)))
        control_run_indices: set[int] = set(plot_run_indices)

        for idx in sorted(path_run_indices):
            env_paths_by_run.setdefault(int(idx), {})

        # 可选：采样固定的 (start, goal) 对集合，用于公平的随机起点/终点评估。
        reset_options_list: list[dict[str, object] | None] = [None] * int(max(0, int(args.runs)))
        # 已移除 precomputed_hybrid_paths，Hybrid A* 评估阶段始终重新规划
        plot_start_xy = tuple(spec.start_xy)
        plot_goal_xy = tuple(spec.goal_xy)

        # --load-pairs：加载预存的 (start, goal) 对，跳过随机采样。
        _load_pairs_path = getattr(args, "load_pairs", None)
        if _load_pairs_path is not None and isinstance(env, UGVBicycleEnv):
            _lp = Path(_load_pairs_path)
            if not _lp.exists():
                raise SystemExit(f"--load-pairs file not found: {_lp}")
            with open(_lp, "r", encoding="utf-8") as _f:
                _pairs_data = json.load(_f)
            _loaded = _pairs_data["pairs"]
            reset_options_list = [
                {"start_xy": (int(p["start_xy"][0]), int(p["start_xy"][1])),
                 "goal_xy": (int(p["goal_xy"][0]), int(p["goal_xy"][1]))}
                for p in _loaded
            ]
            args.runs = len(reset_options_list)
            print(f"[load-pairs] Loaded {len(reset_options_list)} pairs from {_lp}")
            if reset_options_list:
                plot_start_xy = tuple(reset_options_list[plot_run_idx]["start_xy"])  # type: ignore[arg-type]
                plot_goal_xy = tuple(reset_options_list[plot_run_idx]["goal_xy"])  # type: ignore[arg-type]

        elif bool(getattr(args, "random_start_goal", False)) and isinstance(env, UGVBicycleEnv) and int(args.runs) > 0:
            rand_max = None if float(rand_max_cost_m) <= 0.0 else float(rand_max_cost_m)
            if plot_run_idx >= int(args.runs):
                raise SystemExit(
                    f"--plot-run-idx={plot_run_idx} must be < --runs={int(args.runs)} when --random-start-goal is enabled."
                )
            reset_options_list = []
            reject_unreachable = bool(getattr(args, "rand_reject_unreachable", False))
            max_attempts = max(1, int(getattr(args, "rand_reject_max_attempts", 5000)))
            if reject_unreachable:
                # 使用独立于参评算法的 bicycle-kinematic 可达性检查，
                # 避免 Hybrid A* 既做筛选器又做参评算法的公平性偏差。
                pass  # env._dist_m 已在 env 初始化时计算

            sample_pbar = None
            if tqdm is not None:
                sample_pbar = tqdm(
                    total=int(args.runs),
                    desc=f"Sample pairs {env_label}",
                    unit="pair",
                    dynamic_ncols=True,
                    leave=False,
                )

            attempts = 0
            try:
                while len(reset_options_list) < int(args.runs) and attempts < max_attempts:
                    env.reset(
                        seed=int(args.seed) + 90_000 + int(attempts),
                        options={
                            "random_start_goal": True,
                            "rand_min_cost_m": float(rand_min_cost_m),
                            "rand_max_cost_m": rand_max,
                            "rand_fixed_prob": float(args.rand_fixed_prob),
                            "rand_tries": int(args.rand_tries),
                        },
                    )

                    start_xy = (int(env.start_xy[0]), int(env.start_xy[1]))
                    goal_xy = (int(env.goal_xy[0]), int(env.goal_xy[1]))

                    accept = True
                    # 当采样约束过于严格时，环境在耗尽 `rand_tries` 后会回退到规范的
                    # (start,goal) 对。这违背了随机对评估的目的，也会破坏 short/long 套件的分离。
                    if float(getattr(args, "rand_fixed_prob", 0.0)) <= 0.0:
                        if start_xy == (int(spec.start_xy[0]), int(spec.start_xy[1])) and goal_xy == (
                            int(spec.goal_xy[0]),
                            int(spec.goal_xy[1]),
                        ):
                            accept = False
                        if accept:
                            cost0 = float(env._cost_to_goal_m[int(start_xy[1]), int(start_xy[0])])
                            if not math.isfinite(cost0):
                                accept = False
                            elif float(cost0) + 1e-6 < float(rand_min_cost_m):
                                accept = False
                            elif rand_max is not None and float(rand_max) > 0.0 and float(cost0) - 1e-6 > float(rand_max):
                                accept = False

                    if accept and reject_unreachable:
                        # 用 bicycle-kinematic 可达性检查替代 Hybrid A* 筛选
                        reachable = check_bicycle_reachable(
                            env._dist_m,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            cell_size_m=float(cell_size_m),
                            goal_tolerance_m=float(env.goal_tolerance_m),
                        )
                        if not reachable:
                            accept = False

                    if accept:
                        opts: dict[str, object] = {"start_xy": start_xy, "goal_xy": goal_xy}
                        reset_options_list.append(opts)
                        if sample_pbar is not None:
                            sample_pbar.update(1)

                    attempts += 1
                    if sample_pbar is not None and (attempts % 25 == 0):
                        sample_pbar.set_postfix_str(f"attempts={attempts}")
            finally:
                if sample_pbar is not None:
                    sample_pbar.close()

            if len(reset_options_list) < int(args.runs):
                raise RuntimeError(
                    f"Could not sample {int(args.runs)} reachable random (start,goal) pairs for {env_name!r} "
                    f"after {attempts} attempts (rand_min_cost_m={float(rand_min_cost_m):.2f}, rand_max_cost_m={rand_max}). "
                    "Try increasing --rand-tries, adjusting the cost bounds, or disabling screening via --no-rand-reject-unreachable."
                )
            if reset_options_list:
                plot_start_xy = tuple(reset_options_list[plot_run_idx]["start_xy"])  # type: ignore[arg-type]
                plot_goal_xy = tuple(reset_options_list[plot_run_idx]["goal_xy"])  # type: ignore[arg-type]

        use_random_pairs = bool(getattr(args, "random_start_goal", False)) and bool(reset_options_list) and reset_options_list[0] is not None
        env_pbar = None
        if tqdm is not None:
            total_rollouts = 0
            if not bool(args.skip_rl):
                total_rollouts += int(args.runs) * int(len(args.rl_algos))
            if "hybrid_astar" in baselines:
                total_rollouts += int(args.runs) if use_random_pairs else 1
            if "rrt_star" in baselines:
                total_rollouts += int(args.runs)
            if "lo_hybrid_astar" in baselines:
                total_rollouts += int(args.runs) if use_random_pairs else 1
            if total_rollouts > 0:
                env_pbar = tqdm(
                    total=int(total_rollouts),
                    desc=f"Infer {env_label}",
                    unit="rollout",
                    dynamic_ncols=True,
                    leave=True,
                )

        meta_run_indices = sorted(path_run_indices)
        if not reset_options_list or reset_options_list[0] is None:
            panel_start_goal: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
                int(idx): ((int(spec.start_xy[0]), int(spec.start_xy[1])), (int(spec.goal_xy[0]), int(spec.goal_xy[1])))
                for idx in meta_run_indices
            }
        else:
            panel_start_goal = {
                int(idx): (
                    tuple(reset_options_list[int(idx)]["start_xy"]),  # type: ignore[arg-type]
                    tuple(reset_options_list[int(idx)]["goal_xy"]),  # type: ignore[arg-type]
                )
                for idx in meta_run_indices
            }

        for idx, (sp_xy, gp_xy) in panel_start_goal.items():
            meta = dict(base_meta)
            meta["plot_start_x"] = float(sp_xy[0])
            meta["plot_start_y"] = float(sp_xy[1])
            meta["plot_goal_x"] = float(gp_xy[0])
            meta["plot_goal_y"] = float(gp_xy[1])
            meta["plot_run_idx"] = float(idx)
            plot_meta[(env_name, int(idx))] = meta

        if not bool(args.skip_rl):
            # 加载已训练模型
            env_obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)
            agent_cfg = AgentConfig()

            algo_label = {
                "mlp-dqn": "MLP-DQN",
                "mlp-ddqn": "MLP-DDQN",
                "mlp-pddqn": "MLP-PDDQN",
                "cnn-dqn": "CNN-DQN",
                "cnn-ddqn": "CNN-DDQN",
                "cnn-pddqn": "CNN-PDDQN",
            }
            algo_seed_offset = {
                "mlp-dqn": 20_000,
                "mlp-ddqn": 30_000,
                "mlp-pddqn": 60_000,
                "cnn-dqn": 40_000,
                "cnn-ddqn": 50_000,
                "cnn-pddqn": 70_000,
            }

            def resolve_model_path(algo: str) -> Path:
                p = models_dir / env_base / f"{algo}.pt"
                if p.exists():
                    return p
                legacy = {
                    "mlp-dqn": "dqn",
                    "mlp-ddqn": "ddqn",
                    # 向后兼容：旧版 runs 将 Polyak-DDQN 保存为 iddqn/cnn-iddqn。
                    "mlp-pddqn": "iddqn",
                    "cnn-pddqn": "cnn-iddqn",
                }.get(str(algo))
                if legacy is not None:
                    p_legacy = models_dir / env_base / f"{legacy}.pt"
                    if p_legacy.exists():
                        return p_legacy
                return p

            algo_paths = {str(a): resolve_model_path(str(a)) for a in args.rl_algos}
            missing = [str(p) for p in algo_paths.values() if not p.exists()]
            if missing:
                exp = ", ".join(str(p) for p in algo_paths.values())
                raise FileNotFoundError(
                    f"Missing model(s) for env {env_base!r}. Expected: {exp}. "
                    "Point --models at a training run (or an experiment name/dir with a latest run)."
                )

            # 每种架构（MLP / CNN）可能有不同的有效 obs_dim
            # （MLP 会去除 EDT 通道）。智能体构造函数和 load()
            # 会自动处理，所以我们直接传入 env_obs_dim。
            obs_dim = env_obs_dim
            obs_transform = None

            agents: dict[str, DQNFamilyAgent] = {}
            for algo, path in algo_paths.items():
                a = DQNFamilyAgent(str(algo), obs_dim, n_actions, config=agent_cfg, seed=args.seed, device=device, cnn_drop_edt=bool(getattr(args, "cnn_drop_edt", False)))
                a.load(path)
                agents[str(algo)] = a

            for algo in args.rl_algos:
                algo_key = str(algo)
                pretty = algo_label.get(algo_key, algo_key.upper())
                # 当加载的 checkpoint 使用 dueling 头时追加 +Duel 后缀。
                if agents.get(algo_key) and getattr(agents[algo_key], "_net_kwargs", {}).get("dueling", False):
                    pretty = pretty + "+Duel"
                seed_base = int(algo_seed_offset.get(algo_key, 30_000))

                algo_kpis: list[KPI] = []
                algo_times: list[float] = []
                algo_plan_times: list[float] = []
                algo_track_times: list[float] = []
                algo_success = 0
                _use_mpc = bool(getattr(args, "rl_mpc_track", False)) and isinstance(env, UGVBicycleEnv)
                # 当启用 --rl-mpc-track 时：同时累积 "+MPC" 变体
                mpc_kpis: list[KPI] = []
                mpc_times: list[float] = []
                mpc_plan_times: list[float] = []
                mpc_track_times: list[float] = []
                mpc_success = 0
                pretty_mpc = pretty + "+MPC"
                for i in range(int(args.runs)):
                    # --- RL 直接控制（始终执行） ---
                    roll = rollout_agent(
                        env,
                        agents[algo_key],
                        max_steps=args.max_steps,
                        seed=int(args.seed) + seed_base + int(i),
                        reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                        time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                        obs_transform=obs_transform,
                        forest_adm_horizon=int(args.forest_adm_horizon),
                        forest_topk=int(args.forest_topk),
                        forest_min_od_m=float(args.forest_min_od_m),
                        forest_min_progress_m=float(args.forest_min_progress_m),
                        collect_controls=bool(int(i) in control_run_indices),
                        collect_trace=_save_traces,
                    )
                    algo_times.append(float(roll.compute_time_s))
                    algo_plan_times.append(float(roll.compute_time_s))
                    algo_track_times.append(0.0)
                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)][pretty] = PathTrace(path_xy_cells=roll.path_xy_cells, success=bool(roll.reached))
                    if roll.controls is not None and int(i) in control_run_indices:
                        controls_for_plot.setdefault((env_name, int(i)), {})[str(pretty)] = roll.controls

                    start_xy = (int(spec.start_xy[0]), int(spec.start_xy[1]))
                    goal_xy = (int(spec.goal_xy[0]), int(spec.goal_xy[1]))
                    opts = reset_options_list[i] if i < len(reset_options_list) else None
                    if isinstance(opts, dict) and "start_xy" in opts and "goal_xy" in opts:
                        sx, sy = opts["start_xy"]  # type: ignore[misc]
                        gx, gy = opts["goal_xy"]  # type: ignore[misc]
                        start_xy = (int(sx), int(sy))
                        goal_xy = (int(gx), int(gy))

                    # --save-traces：写入 RL 直接控制轨迹 CSV + JSON
                    if _save_traces and roll.trace_rows is not None:
                        _tr_dir = out_dir / "traces"
                        _tr_dir.mkdir(parents=True, exist_ok=True)
                        _csv_name = f"{_safe_slug(env_case)}__{_safe_slug(pretty)}__run{int(i)}.csv"
                        pd.DataFrame(roll.trace_rows).to_csv(_tr_dir / _csv_name, index=False)
                        _save_trace_json(
                            _tr_dir, _csv_name,
                            algorithm=str(pretty), cell_size_m=float(cell_size_m),
                            env_base=str(env_base), env_case=str(env_case),
                            start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                        )

                    raw_corners = float(num_path_corners(roll.path_xy_cells, angle_threshold_deg=13.0))
                    smoothed = smooth_path(roll.path_xy_cells, iterations=2)
                    smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                    run_kpi = KPI(
                        avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                        path_time_s=float(roll.path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                        planning_time_s=float(roll.compute_time_s),
                        tracking_time_s=0.0,
                        inference_time_s=float(roll.compute_time_s),
                        num_corners=raw_corners,
                        max_corner_deg=float(max_corner_degree(smoothed)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": str(pretty),
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(roll.reached) else 0.0,
                            **dict(run_kpi.__dict__),
                        }
                    )
                    if bool(roll.reached):
                        algo_success += 1
                        algo_kpis.append(run_kpi)
                    if env_pbar is not None:
                        env_pbar.set_postfix_str(f"{pretty} run {int(i) + 1}/{int(args.runs)}")
                        env_pbar.update(1)

                    # --- RL+MPC 变体（当启用 --rl-mpc-track 时） ---
                    if _use_mpc:
                        mpc_roll = rollout_agent_plan_then_track(
                            env,
                            agents[algo_key],
                            max_steps=args.max_steps,
                            seed=int(args.seed) + seed_base + int(i),
                            reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                            time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                            obs_transform=obs_transform,
                            forest_adm_horizon=int(args.forest_adm_horizon),
                            forest_topk=int(args.forest_topk),
                            forest_min_od_m=float(args.forest_min_od_m),
                            forest_min_progress_m=float(args.forest_min_progress_m),
                            collect_controls=bool(int(i) in control_run_indices),
                            mpc_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                        )
                        mpc_times.append(float(mpc_roll.compute_time_s))
                        mpc_plan_times.append(float(mpc_roll.planning_time_s) if mpc_roll.planning_time_s else float(mpc_roll.compute_time_s))
                        mpc_track_times.append(float(mpc_roll.tracking_time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)][pretty_mpc] = PathTrace(path_xy_cells=mpc_roll.path_xy_cells, success=bool(mpc_roll.reached))
                        if mpc_roll.controls is not None and int(i) in control_run_indices:
                            controls_for_plot.setdefault((env_name, int(i)), {})[str(pretty_mpc)] = mpc_roll.controls

                        mpc_smoothed = smooth_path(mpc_roll.path_xy_cells, iterations=2)
                        mpc_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in mpc_smoothed]
                        mpc_run_kpi = KPI(
                            avg_path_length=float(path_length(mpc_smoothed)) * float(cell_size_m),
                            path_time_s=float(mpc_roll.path_time_s),
                            avg_curvature_1_m=float(avg_abs_curvature(mpc_smoothed_m)),
                            planning_time_s=float(mpc_roll.planning_time_s) if mpc_roll.planning_time_s else float(mpc_roll.compute_time_s),
                            tracking_time_s=float(mpc_roll.tracking_time_s),
                            inference_time_s=float(mpc_roll.compute_time_s),
                            num_corners=float(num_path_corners(mpc_roll.path_xy_cells, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(mpc_smoothed)),
                        )
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": str(pretty_mpc),
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if bool(mpc_roll.reached) else 0.0,
                                **dict(mpc_run_kpi.__dict__),
                            }
                        )
                        if bool(mpc_roll.reached):
                            mpc_success += 1
                            mpc_kpis.append(mpc_run_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"{pretty_mpc} run {int(i) + 1}/{int(args.runs)}")
                            env_pbar.update(1)

                k = mean_kpi(algo_kpis)
                k_dict = dict(k.__dict__)
                if algo_times:
                    k_dict["planning_time_s"] = float(np.mean(algo_plan_times))
                    k_dict["tracking_time_s"] = float(np.mean(algo_track_times))
                    k_dict["inference_time_s"] = float(np.mean(algo_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": str(pretty),
                        "success_rate": float(algo_success) / float(max(1, int(args.runs))),
                        **k_dict,
                    }
                )

                # 追加 RL+MPC 均值行
                if _use_mpc:
                    mk = mean_kpi(mpc_kpis)
                    mk_dict = dict(mk.__dict__)
                    if mpc_times:
                        mk_dict["planning_time_s"] = float(np.mean(mpc_plan_times))
                        mk_dict["tracking_time_s"] = float(np.mean(mpc_track_times))
                        mk_dict["inference_time_s"] = float(np.mean(mpc_times))
                    rows.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": str(pretty_mpc),
                            "success_rate": float(mpc_success) / float(max(1, int(args.runs))),
                            **mk_dict,
                        }
                    )

        # =====================================================================
        # Expert baseline（cost-to-go 贪心专家）
        # =====================================================================
        if bool(getattr(args, "expert_baseline", False)) and isinstance(env, UGVBicycleEnv):
            expert_pretty = "Expert (CTG)"
            expert_kpis: list[KPI] = []
            expert_times: list[float] = []
            expert_success = 0
            expert_h = int(getattr(args, "expert_horizon", 15))

            for i in range(int(args.runs)):
                roll = rollout_expert(
                    env,
                    max_steps=args.max_steps,
                    seed=int(args.seed) + 80_000 + int(i),
                    reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                    horizon_steps=expert_h,
                    collect_controls=bool(int(i) in control_run_indices),
                    collect_trace=_save_traces,
                )
                expert_times.append(float(roll.compute_time_s))
                if int(i) in path_run_indices:
                    env_paths_by_run[int(i)][expert_pretty] = PathTrace(path_xy_cells=roll.path_xy_cells, success=bool(roll.reached))
                if roll.controls is not None and int(i) in control_run_indices:
                    controls_for_plot.setdefault((env_name, int(i)), {})[str(expert_pretty)] = roll.controls

                start_xy = (int(spec.start_xy[0]), int(spec.start_xy[1]))
                goal_xy = (int(spec.goal_xy[0]), int(spec.goal_xy[1]))
                opts = reset_options_list[i] if i < len(reset_options_list) else None
                if isinstance(opts, dict) and "start_xy" in opts and "goal_xy" in opts:
                    sx, sy = opts["start_xy"]  # type: ignore[misc]
                    gx, gy = opts["goal_xy"]  # type: ignore[misc]
                    start_xy = (int(sx), int(sy))
                    goal_xy = (int(gx), int(gy))

                # --save-traces：写入专家轨迹 CSV + JSON
                if _save_traces and roll.trace_rows is not None:
                    _tr_dir = out_dir / "traces"
                    _tr_dir.mkdir(parents=True, exist_ok=True)
                    _csv_name = f"{_safe_slug(env_case)}__{_safe_slug(expert_pretty)}__run{int(i)}.csv"
                    pd.DataFrame(roll.trace_rows).to_csv(_tr_dir / _csv_name, index=False)
                    _save_trace_json(
                        _tr_dir, _csv_name,
                        algorithm=str(expert_pretty), cell_size_m=float(cell_size_m),
                        env_base=str(env_base), env_case=str(env_case),
                        start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                    )

                raw_corners = float(num_path_corners(roll.path_xy_cells, angle_threshold_deg=13.0))
                smoothed = smooth_path(roll.path_xy_cells, iterations=2)
                smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                run_kpi = KPI(
                    avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                    path_time_s=float(roll.path_time_s),
                    avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                    planning_time_s=float(roll.compute_time_s),
                    tracking_time_s=0.0,
                    inference_time_s=float(roll.compute_time_s),
                    num_corners=raw_corners,
                    max_corner_deg=float(max_corner_degree(smoothed)),
                )
                rows_runs.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": str(expert_pretty),
                        "run_idx": int(i),
                        "start_x": int(start_xy[0]),
                        "start_y": int(start_xy[1]),
                        "goal_x": int(goal_xy[0]),
                        "goal_y": int(goal_xy[1]),
                        "success_rate": 1.0 if bool(roll.reached) else 0.0,
                        **dict(run_kpi.__dict__),
                    }
                )
                if bool(roll.reached):
                    expert_success += 1
                    expert_kpis.append(run_kpi)
                if env_pbar is not None:
                    env_pbar.set_postfix_str(f"{expert_pretty} run {int(i) + 1}/{int(args.runs)}")
                    env_pbar.update(1)

            k = mean_kpi(expert_kpis)
            k_dict = dict(k.__dict__)
            if expert_times:
                k_dict["planning_time_s"] = float(np.mean(expert_times))
                k_dict["tracking_time_s"] = 0.0
                k_dict["inference_time_s"] = float(np.mean(expert_times))
            rows.append(
                {
                    "Environment": str(env_label),
                    "Algorithm": str(expert_pretty),
                    "success_rate": float(expert_success) / float(max(1, int(args.runs))),
                    **k_dict,
                }
            )

        if baselines:
            grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(cell_size_m))
            params = default_ackermann_params()
            if (env_base in FOREST_ENV_ORDER or env_base in REALMAP_ENV_ORDER) and isinstance(env, UGVBicycleEnv):
                footprint = forest_two_circle_footprint()
                goal_xy_tol_m = float(env.goal_tolerance_m)
                goal_theta_tol_rad = float(env.goal_angle_tolerance_rad)
                start_theta_rad = None
            else:
                footprint = point_footprint(cell_size_m=float(cell_size_m))
                goal_xy_tol_m = float(cell_size_m) * 0.5
                goal_theta_tol_rad = float(math.pi)
                start_theta_rad = 0.0

            # 基线使用与 DRL 完全相同的 EDT 碰撞检测。
            _edt_margin = getattr(args, "edt_collision_margin", "diag")
            from ugv_dqn.env import compute_edt_distance_m
            from ugv_dqn.third_party.pathplan.geometry import EDTCollisionChecker
            _edt_dist_m = compute_edt_distance_m(spec.obstacle_grid().astype(np.uint8), cell_size_m=cell_size_m)
            _baseline_edt_checker = EDTCollisionChecker(
                edt_dist_m=_edt_dist_m,
                cell_size_m=cell_size_m,
                footprint=footprint,
                edt_collision_margin=_edt_margin,
            )

            def pair_for_run(i: int) -> tuple[tuple[int, int], tuple[int, int], dict[str, object] | None]:
                if use_random_pairs and i < len(reset_options_list) and reset_options_list[i] is not None:
                    opts = reset_options_list[i] or {}
                    sx, sy = opts["start_xy"]  # type: ignore[misc]
                    gx, gy = opts["goal_xy"]  # type: ignore[misc]
                    return (int(sx), int(sy)), (int(gx), int(gy)), opts
                return (int(spec.start_xy[0]), int(spec.start_xy[1])), (int(spec.goal_xy[0]), int(spec.goal_xy[1])), None

            if "hybrid_astar" in baselines:
                ha_kpis: list[KPI] = []
                ha_plan_times: list[float] = []
                ha_track_times: list[float] = []
                ha_total_times: list[float] = []
                ha_success = 0
                _ha_split = bool(getattr(args, "rl_mpc_track", False)) and isinstance(env, UGVBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False))
                # 拆分模式时：分别累积仅规划和规划+MPC 的指标
                ha_mpc_kpis: list[KPI] = []
                ha_mpc_plan_times: list[float] = []
                ha_mpc_track_times: list[float] = []
                ha_mpc_total_times: list[float] = []
                ha_mpc_success = 0

                n_runs = int(args.runs) if use_random_pairs else 1
                for i in range(n_runs):
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    # 始终重新规划，不复用缓存，确保公平计时和 success rate
                    res = plan_hybrid_astar(
                        grid_map=grid_map,
                        footprint=footprint,
                        params=params,
                        start_xy=start_xy,
                        goal_xy=goal_xy,
                        goal_theta_rad=0.0,
                        start_theta_rad=start_theta_rad,
                        goal_xy_tol_m=goal_xy_tol_m,
                        goal_theta_tol_rad=goal_theta_tol_rad,
                        timeout_s=float(args.baseline_timeout),
                        max_nodes=int(args.hybrid_max_nodes),
                        collision_checker=_baseline_edt_checker,
                        smooth=bool(getattr(args, "ha_smooth", 1)),
                        rs_heuristic_max_dist=float(getattr(args, "ha_rs_heuristic_max_dist", 15.0)),
                        xy_resolution=float(getattr(args, "ha_xy_resolution", 0.0)),
                        step_length=float(getattr(args, "ha_step_length", 0.3)),
                    )

                    if _ha_split:
                        # --- 仅规划行 ("Hybrid A*") ---
                        plan_path = list(res.path_xy_cells)
                        plan_reached = bool(res.success)
                        plan_smoothed = smooth_path(plan_path, iterations=2)
                        plan_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in plan_smoothed]
                        plan_path_time = float(path_length(plan_smoothed_m)) / max(1e-9, float(env.model.v_max_m_s)) if isinstance(env, UGVBicycleEnv) else 0.0
                        plan_kpi = KPI(
                            avg_path_length=float(path_length(plan_smoothed)) * float(cell_size_m),
                            path_time_s=plan_path_time,
                            avg_curvature_1_m=float(avg_abs_curvature(plan_smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=0.0,
                            inference_time_s=float(res.time_s),
                            num_corners=float(num_path_corners(plan_path, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(plan_smoothed)),
                        )
                        ha_plan_times.append(float(res.time_s))
                        ha_track_times.append(0.0)
                        ha_total_times.append(float(res.time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["Hybrid A*"] = PathTrace(path_xy_cells=plan_path, success=plan_reached)
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "Hybrid A*",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if plan_reached else 0.0,
                                **dict(plan_kpi.__dict__),
                            }
                        )
                        if plan_reached and plan_path:
                            ha_success += 1
                            ha_kpis.append(plan_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"Hybrid A* run {int(i) + 1}/{int(n_runs)}")
                            env_pbar.update(1)

                        # --- 规划+MPC 行 ("Hybrid A*+MPC") ---
                        if bool(res.success):
                            trace_path = None
                            if bool(getattr(args, "forest_baseline_save_traces", False)) or _save_traces:
                                trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__Hybrid_A__run{int(i)}.csv"
                            roll = rollout_tracked_path_mpc(
                                env,
                                list(res.path_xy_cells),
                                max_steps=args.max_steps,
                                seed=args.seed + 30_000 + i,
                                reset_options=r_opts,
                                time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                                trace_path=trace_path,
                                n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                                collect_controls=bool(int(i) in control_run_indices),
                            )
                            mpc_exec_path = list(roll.path_xy_cells)
                            mpc_reached = bool(roll.reached)
                            mpc_track_t = float(roll.compute_time_s)
                            mpc_path_t = float(roll.path_time_s)
                            if int(i) in control_run_indices and roll.controls is not None:
                                controls_for_plot.setdefault((env_name, int(i)), {})["Hybrid A*+MPC"] = roll.controls
                            if trace_path is not None and _save_traces:
                                _save_trace_json(
                                    trace_path.parent, trace_path.name,
                                    algorithm="Hybrid A*", cell_size_m=float(cell_size_m),
                                    env_base=str(env_base), env_case=str(env_case),
                                    start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                                )
                        else:
                            mpc_exec_path = list(res.path_xy_cells)
                            mpc_reached = False
                            mpc_track_t = 0.0
                            mpc_path_t = plan_path_time
                        mpc_smoothed = smooth_path(mpc_exec_path, iterations=2)
                        mpc_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in mpc_smoothed]
                        mpc_kpi = KPI(
                            avg_path_length=float(path_length(mpc_smoothed)) * float(cell_size_m),
                            path_time_s=mpc_path_t,
                            avg_curvature_1_m=float(avg_abs_curvature(mpc_smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=mpc_track_t,
                            inference_time_s=float(res.time_s) + mpc_track_t,
                            num_corners=float(num_path_corners(mpc_exec_path, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(mpc_smoothed)),
                        )
                        ha_mpc_plan_times.append(float(res.time_s))
                        ha_mpc_track_times.append(mpc_track_t)
                        ha_mpc_total_times.append(float(res.time_s) + mpc_track_t)
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["Hybrid A*+MPC"] = PathTrace(path_xy_cells=mpc_exec_path, success=mpc_reached)
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "Hybrid A*+MPC",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if mpc_reached else 0.0,
                                **dict(mpc_kpi.__dict__),
                            }
                        )
                        if mpc_reached and mpc_exec_path:
                            ha_mpc_success += 1
                            ha_mpc_kpis.append(mpc_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"Hybrid A*+MPC run {int(i) + 1}/{int(n_runs)}")
                            env_pbar.update(1)

                    else:
                        # --- 原始单行模式（不拆分） ---
                        ha_exec_path = list(res.path_xy_cells)
                        ha_reached = bool(res.success)
                        ha_track_time_s = 0.0
                        ha_path_time_s = float("nan")
                        if bool(res.success) and isinstance(env, UGVBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                            trace_path = None
                            if bool(getattr(args, "forest_baseline_save_traces", False)) or _save_traces:
                                trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__Hybrid_A__run{int(i)}.csv"
                            roll = rollout_tracked_path_mpc(
                                env,
                                ha_exec_path,
                                max_steps=args.max_steps,
                                seed=args.seed + 30_000 + i,
                                reset_options=r_opts,
                                time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                                trace_path=trace_path,
                                n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                                collect_controls=bool(int(i) in control_run_indices),
                            )
                            ha_exec_path = list(roll.path_xy_cells)
                            ha_track_time_s = float(roll.compute_time_s)
                            ha_reached = bool(roll.reached)
                            ha_path_time_s = float(roll.path_time_s)
                            if int(i) in control_run_indices and roll.controls is not None:
                                controls_for_plot.setdefault((env_name, int(i)), {})["Hybrid A*"] = roll.controls
                            if trace_path is not None and _save_traces:
                                _save_trace_json(
                                    trace_path.parent, trace_path.name,
                                    algorithm="Hybrid A*", cell_size_m=float(cell_size_m),
                                    env_base=str(env_base), env_case=str(env_case),
                                    start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                                )

                        ha_plan_times.append(float(res.time_s))
                        ha_track_times.append(float(ha_track_time_s))
                        ha_total_times.append(float(res.time_s) + float(ha_track_time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["Hybrid A*"] = PathTrace(path_xy_cells=ha_exec_path, success=bool(ha_reached))

                        raw_corners = float(num_path_corners(ha_exec_path, angle_threshold_deg=13.0))
                        smoothed = smooth_path(ha_exec_path, iterations=2)
                        smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                        if not math.isfinite(float(ha_path_time_s)) and isinstance(env, UGVBicycleEnv):
                            ha_path_time_s = float(path_length(smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                        run_kpi = KPI(
                            avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                            path_time_s=float(ha_path_time_s),
                            avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=float(ha_track_time_s),
                            inference_time_s=float(res.time_s) + float(ha_track_time_s),
                            num_corners=raw_corners,
                            max_corner_deg=float(max_corner_degree(smoothed)),
                        )
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "Hybrid A*",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if bool(ha_reached) else 0.0,
                                **dict(run_kpi.__dict__),
                            }
                        )
                        if bool(ha_reached) and ha_exec_path:
                            ha_success += 1
                            ha_kpis.append(run_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"Hybrid A* run {int(i) + 1}/{int(n_runs)}")
                            env_pbar.update(1)

                k = mean_kpi(ha_kpis)
                k_dict = dict(k.__dict__)
                if ha_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(ha_plan_times))
                if ha_track_times:
                    k_dict["tracking_time_s"] = float(np.mean(ha_track_times))
                if ha_total_times:
                    k_dict["inference_time_s"] = float(np.mean(ha_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "Hybrid A*",
                        "success_rate": float(ha_success) / float(max(1, int(n_runs))),
                        **k_dict,
                    }
                )
                # 追加 Hybrid A*+MPC 均值行
                if _ha_split:
                    mk = mean_kpi(ha_mpc_kpis)
                    mk_dict = dict(mk.__dict__)
                    if ha_mpc_plan_times:
                        mk_dict["planning_time_s"] = float(np.mean(ha_mpc_plan_times))
                    if ha_mpc_track_times:
                        mk_dict["tracking_time_s"] = float(np.mean(ha_mpc_track_times))
                    if ha_mpc_total_times:
                        mk_dict["inference_time_s"] = float(np.mean(ha_mpc_total_times))
                    rows.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "Hybrid A*+MPC",
                            "success_rate": float(ha_mpc_success) / float(max(1, int(n_runs))),
                            **mk_dict,
                        }
                    )

            if "rrt_star" in baselines:
                rrt_kpis: list[KPI] = []
                rrt_plan_times: list[float] = []
                rrt_track_times: list[float] = []
                rrt_total_times: list[float] = []
                rrt_success = 0
                _rrt_split = bool(getattr(args, "rl_mpc_track", False)) and isinstance(env, UGVBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False))
                rrt_mpc_kpis: list[KPI] = []
                rrt_mpc_plan_times: list[float] = []
                rrt_mpc_track_times: list[float] = []
                rrt_mpc_total_times: list[float] = []
                rrt_mpc_success = 0

                for i in range(args.runs):
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    res = plan_rrt_star(
                        grid_map=grid_map,
                        footprint=footprint,
                        params=params,
                        start_xy=start_xy,
                        goal_xy=goal_xy,
                        goal_theta_rad=0.0,
                        start_theta_rad=start_theta_rad,
                        goal_xy_tol_m=goal_xy_tol_m,
                        goal_theta_tol_rad=goal_theta_tol_rad,
                        timeout_s=float(args.baseline_timeout),
                        max_iter=int(args.rrt_max_iter),
                        seed=args.seed + 30_000 + i,
                        collision_checker=_baseline_edt_checker,
                    )
                    if _rrt_split:
                        # --- 仅规划行 ("RRT*") ---
                        plan_path = list(res.path_xy_cells)
                        plan_reached = bool(res.success)
                        plan_smoothed = smooth_path(plan_path, iterations=2)
                        plan_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in plan_smoothed]
                        plan_path_time = float(path_length(plan_smoothed_m)) / max(1e-9, float(env.model.v_max_m_s)) if isinstance(env, UGVBicycleEnv) else 0.0
                        plan_kpi = KPI(
                            avg_path_length=float(path_length(plan_smoothed)) * float(cell_size_m),
                            path_time_s=plan_path_time,
                            avg_curvature_1_m=float(avg_abs_curvature(plan_smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=0.0,
                            inference_time_s=float(res.time_s),
                            num_corners=float(num_path_corners(plan_path, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(plan_smoothed)),
                        )
                        rrt_plan_times.append(float(res.time_s))
                        rrt_track_times.append(0.0)
                        rrt_total_times.append(float(res.time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["RRT*"] = PathTrace(path_xy_cells=plan_path, success=plan_reached)
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "RRT*",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if plan_reached else 0.0,
                                **dict(plan_kpi.__dict__),
                            }
                        )
                        if plan_reached and plan_path:
                            rrt_success += 1
                            rrt_kpis.append(plan_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"RRT* run {int(i) + 1}/{int(args.runs)}")
                            env_pbar.update(1)

                        # --- 规划+MPC 行 ("RRT*+MPC") ---
                        if bool(res.success):
                            trace_path = None
                            if bool(getattr(args, "forest_baseline_save_traces", False)) or _save_traces:
                                trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__RRT__run{int(i)}.csv"
                            roll = rollout_tracked_path_mpc(
                                env,
                                list(res.path_xy_cells),
                                max_steps=args.max_steps,
                                seed=args.seed + 40_000 + i,
                                reset_options=r_opts,
                                time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                                trace_path=trace_path,
                                n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                                collect_controls=bool(int(i) in control_run_indices),
                            )
                            mpc_exec_path = list(roll.path_xy_cells)
                            mpc_reached = bool(roll.reached)
                            mpc_track_t = float(roll.compute_time_s)
                            mpc_path_t = float(roll.path_time_s)
                            if int(i) in control_run_indices and roll.controls is not None:
                                controls_for_plot.setdefault((env_name, int(i)), {})["RRT*+MPC"] = roll.controls
                            if trace_path is not None and _save_traces:
                                _save_trace_json(
                                    trace_path.parent, trace_path.name,
                                    algorithm="SS-RRT*", cell_size_m=float(cell_size_m),
                                    env_base=str(env_base), env_case=str(env_case),
                                    start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                                )
                        else:
                            mpc_exec_path = list(res.path_xy_cells)
                            mpc_reached = False
                            mpc_track_t = 0.0
                            mpc_path_t = plan_path_time
                        mpc_smoothed = smooth_path(mpc_exec_path, iterations=2)
                        mpc_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in mpc_smoothed]
                        mpc_kpi = KPI(
                            avg_path_length=float(path_length(mpc_smoothed)) * float(cell_size_m),
                            path_time_s=mpc_path_t,
                            avg_curvature_1_m=float(avg_abs_curvature(mpc_smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=mpc_track_t,
                            inference_time_s=float(res.time_s) + mpc_track_t,
                            num_corners=float(num_path_corners(mpc_exec_path, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(mpc_smoothed)),
                        )
                        rrt_mpc_plan_times.append(float(res.time_s))
                        rrt_mpc_track_times.append(mpc_track_t)
                        rrt_mpc_total_times.append(float(res.time_s) + mpc_track_t)
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["RRT*+MPC"] = PathTrace(path_xy_cells=mpc_exec_path, success=mpc_reached)
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "RRT*+MPC",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if mpc_reached else 0.0,
                                **dict(mpc_kpi.__dict__),
                            }
                        )
                        if mpc_reached and mpc_exec_path:
                            rrt_mpc_success += 1
                            rrt_mpc_kpis.append(mpc_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"RRT*+MPC run {int(i) + 1}/{int(args.runs)}")
                            env_pbar.update(1)

                    else:
                        # --- 原始单行模式（不拆分） ---
                        exec_path = list(res.path_xy_cells)
                        reached = bool(res.success)
                        track_time_s = 0.0
                        path_time_s = float("nan")
                        if bool(res.success) and isinstance(env, UGVBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                            trace_path = None
                            if bool(getattr(args, "forest_baseline_save_traces", False)) or _save_traces:
                                trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__RRT__run{int(i)}.csv"
                            roll = rollout_tracked_path_mpc(
                                env,
                                exec_path,
                                max_steps=args.max_steps,
                                seed=args.seed + 40_000 + i,
                                reset_options=r_opts,
                                time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                                trace_path=trace_path,
                                n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                                collect_controls=bool(int(i) in control_run_indices),
                            )
                            exec_path = list(roll.path_xy_cells)
                            track_time_s = float(roll.compute_time_s)
                            reached = bool(roll.reached)
                            path_time_s = float(roll.path_time_s)
                            if int(i) in control_run_indices and roll.controls is not None:
                                controls_for_plot.setdefault((env_name, int(i)), {})["RRT*"] = roll.controls
                            if trace_path is not None and _save_traces:
                                _save_trace_json(
                                    trace_path.parent, trace_path.name,
                                    algorithm="SS-RRT*", cell_size_m=float(cell_size_m),
                                    env_base=str(env_base), env_case=str(env_case),
                                    start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                                )

                        rrt_plan_times.append(float(res.time_s))
                        rrt_track_times.append(float(track_time_s))
                        rrt_total_times.append(float(res.time_s) + float(track_time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["RRT*"] = PathTrace(path_xy_cells=exec_path, success=bool(reached))

                        raw_corners = float(num_path_corners(exec_path, angle_threshold_deg=13.0))
                        smoothed = smooth_path(exec_path, iterations=2)
                        smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                        if not math.isfinite(float(path_time_s)) and isinstance(env, UGVBicycleEnv):
                            path_time_s = float(path_length(smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                        run_kpi = KPI(
                            avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                            path_time_s=float(path_time_s),
                            avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=float(track_time_s),
                            inference_time_s=float(res.time_s) + float(track_time_s),
                            num_corners=raw_corners,
                            max_corner_deg=float(max_corner_degree(smoothed)),
                        )
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "RRT*",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if bool(reached) else 0.0,
                                **dict(run_kpi.__dict__),
                            }
                        )
                        if bool(reached) and exec_path:
                            rrt_success += 1
                            rrt_kpis.append(run_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"RRT* run {int(i) + 1}/{int(args.runs)}")
                            env_pbar.update(1)

                k = mean_kpi(rrt_kpis)
                k_dict = dict(k.__dict__)
                if rrt_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(rrt_plan_times))
                if rrt_track_times:
                    k_dict["tracking_time_s"] = float(np.mean(rrt_track_times))
                if rrt_total_times:
                    k_dict["inference_time_s"] = float(np.mean(rrt_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "RRT*",
                        "success_rate": float(rrt_success) / float(max(1, int(args.runs))),
                        **k_dict,
                    }
                )
                # 追加 RRT*+MPC 均值行
                if _rrt_split:
                    mk = mean_kpi(rrt_mpc_kpis)
                    mk_dict = dict(mk.__dict__)
                    if rrt_mpc_plan_times:
                        mk_dict["planning_time_s"] = float(np.mean(rrt_mpc_plan_times))
                    if rrt_mpc_track_times:
                        mk_dict["tracking_time_s"] = float(np.mean(rrt_mpc_track_times))
                    if rrt_mpc_total_times:
                        mk_dict["inference_time_s"] = float(np.mean(rrt_mpc_total_times))
                    rows.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "RRT*+MPC",
                            "success_rate": float(rrt_mpc_success) / float(max(1, int(args.runs))),
                            **mk_dict,
                        }
                    )

            if "lo_hybrid_astar" in baselines:
                loha_kpis: list[KPI] = []
                loha_plan_times: list[float] = []
                loha_track_times: list[float] = []
                loha_total_times: list[float] = []
                loha_success = 0
                _loha_split = bool(getattr(args, "rl_mpc_track", False)) and isinstance(env, UGVBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False))
                loha_mpc_kpis: list[KPI] = []
                loha_mpc_plan_times: list[float] = []
                loha_mpc_track_times: list[float] = []
                loha_mpc_total_times: list[float] = []
                loha_mpc_success = 0

                n_runs = int(args.runs) if use_random_pairs else 1
                for i in range(n_runs):
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    res = plan_lo_hybrid_astar(
                        grid_map=grid_map,
                        footprint=footprint,
                        params=params,
                        start_xy=start_xy,
                        goal_xy=goal_xy,
                        goal_theta_rad=0.0,
                        start_theta_rad=start_theta_rad,
                        goal_xy_tol_m=goal_xy_tol_m,
                        goal_theta_tol_rad=goal_theta_tol_rad,
                        timeout_s=float(args.baseline_timeout),
                        max_nodes=int(args.hybrid_max_nodes),
                        lo_iterations=int(getattr(args, "loha_lo_iterations", 0)),
                        collision_checker=_baseline_edt_checker,
                    )

                    if _loha_split:
                        # --- 仅规划行 ("LO-HA*") ---
                        plan_path = list(res.path_xy_cells)
                        plan_reached = bool(res.success)
                        plan_smoothed = smooth_path(plan_path, iterations=2)
                        plan_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in plan_smoothed]
                        plan_path_time = float(path_length(plan_smoothed_m)) / max(1e-9, float(env.model.v_max_m_s)) if isinstance(env, UGVBicycleEnv) else 0.0
                        plan_kpi = KPI(
                            avg_path_length=float(path_length(plan_smoothed)) * float(cell_size_m),
                            path_time_s=plan_path_time,
                            avg_curvature_1_m=float(avg_abs_curvature(plan_smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=0.0,
                            inference_time_s=float(res.time_s),
                            num_corners=float(num_path_corners(plan_path, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(plan_smoothed)),
                        )
                        loha_plan_times.append(float(res.time_s))
                        loha_track_times.append(0.0)
                        loha_total_times.append(float(res.time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["LO-HA*"] = PathTrace(path_xy_cells=plan_path, success=plan_reached)
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "LO-HA*",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if plan_reached else 0.0,
                                **dict(plan_kpi.__dict__),
                            }
                        )
                        if plan_reached and plan_path:
                            loha_success += 1
                            loha_kpis.append(plan_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"LO-HA* run {int(i) + 1}/{int(n_runs)}")
                            env_pbar.update(1)

                        # --- 规划+MPC 行 ("LO-HA*+MPC") ---
                        if bool(res.success):
                            trace_path = None
                            if bool(getattr(args, "forest_baseline_save_traces", False)) or _save_traces:
                                trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__LO_HA__run{int(i)}.csv"
                            roll = rollout_tracked_path_mpc(
                                env,
                                list(res.path_xy_cells),
                                max_steps=args.max_steps,
                                seed=args.seed + 50_000 + i,
                                reset_options=r_opts,
                                time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                                trace_path=trace_path,
                                n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                                collect_controls=bool(int(i) in control_run_indices),
                            )
                            mpc_exec_path = list(roll.path_xy_cells)
                            mpc_reached = bool(roll.reached)
                            mpc_track_t = float(roll.compute_time_s)
                            mpc_path_t = float(roll.path_time_s)
                            if int(i) in control_run_indices and roll.controls is not None:
                                controls_for_plot.setdefault((env_name, int(i)), {})["LO-HA*+MPC"] = roll.controls
                            if trace_path is not None and _save_traces:
                                _save_trace_json(
                                    trace_path.parent, trace_path.name,
                                    algorithm="LO-HA*", cell_size_m=float(cell_size_m),
                                    env_base=str(env_base), env_case=str(env_case),
                                    start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                                )
                        else:
                            mpc_exec_path = list(res.path_xy_cells)
                            mpc_reached = False
                            mpc_track_t = 0.0
                            mpc_path_t = plan_path_time
                        mpc_smoothed = smooth_path(mpc_exec_path, iterations=2)
                        mpc_smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in mpc_smoothed]
                        mpc_kpi = KPI(
                            avg_path_length=float(path_length(mpc_smoothed)) * float(cell_size_m),
                            path_time_s=mpc_path_t,
                            avg_curvature_1_m=float(avg_abs_curvature(mpc_smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=mpc_track_t,
                            inference_time_s=float(res.time_s) + mpc_track_t,
                            num_corners=float(num_path_corners(mpc_exec_path, angle_threshold_deg=13.0)),
                            max_corner_deg=float(max_corner_degree(mpc_smoothed)),
                        )
                        loha_mpc_plan_times.append(float(res.time_s))
                        loha_mpc_track_times.append(mpc_track_t)
                        loha_mpc_total_times.append(float(res.time_s) + mpc_track_t)
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["LO-HA*+MPC"] = PathTrace(path_xy_cells=mpc_exec_path, success=mpc_reached)
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "LO-HA*+MPC",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if mpc_reached else 0.0,
                                **dict(mpc_kpi.__dict__),
                            }
                        )
                        if mpc_reached and mpc_exec_path:
                            loha_mpc_success += 1
                            loha_mpc_kpis.append(mpc_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"LO-HA*+MPC run {int(i) + 1}/{int(n_runs)}")
                            env_pbar.update(1)

                    else:
                        # --- 原始单行模式（不拆分） ---
                        loha_exec_path = list(res.path_xy_cells)
                        loha_reached = bool(res.success)
                        loha_track_time_s = 0.0
                        loha_path_time_s = float("nan")
                        if bool(res.success) and isinstance(env, UGVBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                            trace_path = None
                            if bool(getattr(args, "forest_baseline_save_traces", False)) or _save_traces:
                                trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__LO_HA__run{int(i)}.csv"
                            roll = rollout_tracked_path_mpc(
                                env,
                                loha_exec_path,
                                max_steps=args.max_steps,
                                seed=args.seed + 50_000 + i,
                                reset_options=r_opts,
                                time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                                trace_path=trace_path,
                                n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                                collect_controls=bool(int(i) in control_run_indices),
                            )
                            loha_exec_path = list(roll.path_xy_cells)
                            loha_track_time_s = float(roll.compute_time_s)
                            loha_reached = bool(roll.reached)
                            loha_path_time_s = float(roll.path_time_s)
                            if int(i) in control_run_indices and roll.controls is not None:
                                controls_for_plot.setdefault((env_name, int(i)), {})["LO-HA*"] = roll.controls
                            if trace_path is not None and _save_traces:
                                _save_trace_json(
                                    trace_path.parent, trace_path.name,
                                    algorithm="LO-HA*", cell_size_m=float(cell_size_m),
                                    env_base=str(env_base), env_case=str(env_case),
                                    start_xy=start_xy, goal_xy=goal_xy, run_idx=int(i),
                                )

                        loha_plan_times.append(float(res.time_s))
                        loha_track_times.append(float(loha_track_time_s))
                        loha_total_times.append(float(res.time_s) + float(loha_track_time_s))
                        if int(i) in path_run_indices:
                            env_paths_by_run[int(i)]["LO-HA*"] = PathTrace(path_xy_cells=loha_exec_path, success=bool(loha_reached))

                        raw_corners = float(num_path_corners(loha_exec_path, angle_threshold_deg=13.0))
                        smoothed = smooth_path(loha_exec_path, iterations=2)
                        smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                        if not math.isfinite(float(loha_path_time_s)) and isinstance(env, UGVBicycleEnv):
                            loha_path_time_s = float(path_length(smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                        run_kpi = KPI(
                            avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                            path_time_s=float(loha_path_time_s),
                            avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                            planning_time_s=float(res.time_s),
                            tracking_time_s=float(loha_track_time_s),
                            inference_time_s=float(res.time_s) + float(loha_track_time_s),
                            num_corners=raw_corners,
                            max_corner_deg=float(max_corner_degree(smoothed)),
                        )
                        rows_runs.append(
                            {
                                "Environment": str(env_label),
                                "Algorithm": "LO-HA*",
                                "run_idx": int(i),
                                "start_x": int(start_xy[0]),
                                "start_y": int(start_xy[1]),
                                "goal_x": int(goal_xy[0]),
                                "goal_y": int(goal_xy[1]),
                                "success_rate": 1.0 if bool(loha_reached) else 0.0,
                                **dict(run_kpi.__dict__),
                            }
                        )
                        if bool(loha_reached) and loha_exec_path:
                            loha_success += 1
                            loha_kpis.append(run_kpi)
                        if env_pbar is not None:
                            env_pbar.set_postfix_str(f"LO-HA* run {int(i) + 1}/{int(n_runs)}")
                            env_pbar.update(1)

                k = mean_kpi(loha_kpis)
                k_dict = dict(k.__dict__)
                if loha_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(loha_plan_times))
                if loha_track_times:
                    k_dict["tracking_time_s"] = float(np.mean(loha_track_times))
                if loha_total_times:
                    k_dict["inference_time_s"] = float(np.mean(loha_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "LO-HA*",
                        "success_rate": float(loha_success) / float(max(1, int(n_runs))),
                        **k_dict,
                    }
                )
                # 追加 LO-HA*+MPC 均值行
                if _loha_split:
                    mk = mean_kpi(loha_mpc_kpis)
                    mk_dict = dict(mk.__dict__)
                    if loha_mpc_plan_times:
                        mk_dict["planning_time_s"] = float(np.mean(loha_mpc_plan_times))
                    if loha_mpc_track_times:
                        mk_dict["tracking_time_s"] = float(np.mean(loha_mpc_track_times))
                    if loha_mpc_total_times:
                        mk_dict["inference_time_s"] = float(np.mean(loha_mpc_total_times))
                    rows.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "LO-HA*+MPC",
                            "success_rate": float(loha_mpc_success) / float(max(1, int(n_runs))),
                            **mk_dict,
                        }
                    )

        if env_pbar is not None:
            env_pbar.close()
        for run_idx, run_paths in env_paths_by_run.items():
            paths_for_plot[(env_name, int(run_idx))] = dict(run_paths)

    # ── 保存路径数据：CSV（轨迹点）+ pkl（栅格元数据） ──
    if paths_for_plot:
        import csv as _csv
        import pickle as _pkl

        # 1) paths_all.csv —— 每个坐标点一行，可直接画图
        _csv_path = out_dir / "paths_all.csv"
        with open(_csv_path, "w", newline="", encoding="utf-8") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["env", "run_idx", "algo", "point_idx", "x_m", "y_m", "success"])
            for (ename, ridx), alg_paths in sorted(paths_for_plot.items()):
                for alg_name, pt in alg_paths.items():
                    _succ = int(pt.success)
                    for pidx, (cx, cy) in enumerate(pt.path_xy_cells):
                        _w.writerow([str(ename), int(ridx), str(alg_name), int(pidx),
                                     round(float(cx) * float(cell_size_m), 6),
                                     round(float(cy) * float(cell_size_m), 6),
                                     _succ])
        print(f"Wrote: {_csv_path}")

        # 2) map_meta.pkl —— 障碍物栅格 + 栅格尺寸（二维数组不适合 CSV）
        _meta_path = out_dir / "map_meta.pkl"
        with open(_meta_path, "wb") as _f:
            _pkl.dump({"cell_size_m": float(cell_size_m), "obstacle_grid": grid}, _f)
        print(f"Wrote: {_meta_path}")

    table = pd.DataFrame(rows_runs)
    # 美化列顺序
    table = table[
        [
            "Environment",
            "Algorithm",
            "run_idx",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
            "success_rate",
            "avg_path_length",
            "path_time_s",
            "avg_curvature_1_m",
            "planning_time_s",
            "tracking_time_s",
            "num_corners",
            "inference_time_s",
            "max_corner_deg",
        ]
    ]
    table = table.copy()

    # 综合指标（越低越好）：结合路径长度和计算时间，
    # 然后通过成功率惩罚未到达行为。
    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table["planning_cost"] = planning_cost

    # 综合评分（越低越好）：结合路径时间、曲率和规划计算时间，
    # 然后通过成功率惩罚未到达行为。
    w_pt = float(getattr(args, "composite_w_path_time", 1.0))
    w_k = float(getattr(args, "composite_w_avg_curvature", 1.0))
    w_pl = float(getattr(args, "composite_w_planning_time", 1.0))
    w_sum = max(1e-12, float(w_pt + w_k + w_pl))

    def _minmax_norm(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").astype(float)
        v = x.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(v)
        if not bool(finite.any()):
            return pd.Series(np.zeros_like(v, dtype=float), index=x.index)
        mn = float(np.min(v[finite]))
        mx = float(np.max(v[finite]))
        d = float(mx - mn)
        if not math.isfinite(d) or d < 1e-12:
            return pd.Series(np.zeros_like(v, dtype=float), index=x.index)
        out = (v - mn) / d
        out = np.where(finite, out, np.nan)
        return pd.Series(out.astype(float, copy=False), index=x.index)

    group_keys = ["Environment", "run_idx"]
    n_pt = table.groupby(group_keys, sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
    n_k = table.groupby(group_keys, sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
    n_pl = table.groupby(group_keys, sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
    base_score = (w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum
    sr_denom2 = sr_raw.clip(lower=1e-6)
    composite_score = (base_score / sr_denom2).astype(float)
    composite_score = composite_score.where(sr_raw > 0.0, other=float("inf"))
    table["composite_score"] = composite_score

    table["success_rate"] = pd.to_numeric(table["success_rate"], errors="coerce").astype(float).round(3)
    table["avg_path_length"] = table["avg_path_length"].astype(float).round(4)
    table["path_time_s"] = pd.to_numeric(table["path_time_s"], errors="coerce").astype(float).round(4)
    table["avg_curvature_1_m"] = pd.to_numeric(table["avg_curvature_1_m"], errors="coerce").astype(float).round(6)
    table["planning_time_s"] = pd.to_numeric(table["planning_time_s"], errors="coerce").astype(float).round(5)
    table["tracking_time_s"] = pd.to_numeric(table["tracking_time_s"], errors="coerce").astype(float).round(5)
    table["num_corners"] = pd.to_numeric(table["num_corners"], errors="coerce").round(0).astype("Int64")
    table["inference_time_s"] = table["inference_time_s"].astype(float).round(5)
    table["max_corner_deg"] = pd.to_numeric(table["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table["planning_cost"] = pd.to_numeric(table["planning_cost"], errors="coerce").astype(float).round(3)
    table["composite_score"] = pd.to_numeric(table["composite_score"], errors="coerce").astype(float).round(3)
    table.to_csv(out_dir / "table2_kpis_raw.csv", index=False)

    table_pretty = table.rename(
        columns={
            "Algorithm": "Algorithm name",
            "run_idx": "Run index",
            "start_x": "Start x",
            "start_y": "Start y",
            "goal_x": "Goal x",
            "goal_y": "Goal y",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "path_time_s": "Path time (s)",
            "avg_curvature_1_m": "Average curvature (1/m)",
            "planning_time_s": "Planning time (s)",
            "tracking_time_s": "Tracking time (s)",
            "num_corners": "Number of path corners",
            "inference_time_s": "Compute time (s)",
            "max_corner_deg": "Max corner degree (°)",
            "planning_cost": "Planning cost (m)",
            "composite_score": "Composite score",
        }
    )
    table_pretty.to_csv(out_dir / "table2_kpis.csv", index=False)
    table_pretty.to_markdown(out_dir / "table2_kpis.md", index=False)

    # 同时写入均值 KPI 表（之前的默认行为）。
    table_mean = pd.DataFrame(rows)
    table_mean = table_mean[
        [
            "Environment",
            "Algorithm",
            "success_rate",
            "avg_path_length",
            "path_time_s",
            "avg_curvature_1_m",
            "planning_time_s",
            "tracking_time_s",
            "num_corners",
            "inference_time_s",
            "max_corner_deg",
        ]
    ]
    table_mean = table_mean.copy()

    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table_mean["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table_mean["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table_mean["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table_mean["planning_cost"] = planning_cost

    group_keys = ["Environment"]
    n_pt = table_mean.groupby(group_keys, sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
    n_k = table_mean.groupby(group_keys, sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
    n_pl = table_mean.groupby(group_keys, sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
    base_score = (w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum
    sr_denom2 = sr_raw.clip(lower=1e-6)
    composite_score = (base_score / sr_denom2).astype(float)
    composite_score = composite_score.where(sr_raw > 0.0, other=float("inf"))
    table_mean["composite_score"] = composite_score

    table_mean["success_rate"] = pd.to_numeric(table_mean["success_rate"], errors="coerce").astype(float).round(3)
    table_mean["avg_path_length"] = table_mean["avg_path_length"].astype(float).round(4)
    table_mean["path_time_s"] = pd.to_numeric(table_mean["path_time_s"], errors="coerce").astype(float).round(4)
    table_mean["avg_curvature_1_m"] = pd.to_numeric(table_mean["avg_curvature_1_m"], errors="coerce").astype(float).round(6)
    table_mean["planning_time_s"] = pd.to_numeric(table_mean["planning_time_s"], errors="coerce").astype(float).round(5)
    table_mean["tracking_time_s"] = pd.to_numeric(table_mean["tracking_time_s"], errors="coerce").astype(float).round(5)
    table_mean["num_corners"] = pd.to_numeric(table_mean["num_corners"], errors="coerce").round(0).astype("Int64")
    table_mean["inference_time_s"] = table_mean["inference_time_s"].astype(float).round(5)
    table_mean["max_corner_deg"] = pd.to_numeric(table_mean["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table_mean["planning_cost"] = pd.to_numeric(table_mean["planning_cost"], errors="coerce").astype(float).round(3)
    table_mean["composite_score"] = pd.to_numeric(table_mean["composite_score"], errors="coerce").astype(float).round(3)
    table_mean.to_csv(out_dir / "table2_kpis_mean_raw.csv", index=False)

    table_mean_pretty = table_mean.rename(
        columns={
            "Algorithm": "Algorithm name",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "path_time_s": "Path time (s)",
            "avg_curvature_1_m": "Average curvature (1/m)",
            "planning_time_s": "Planning time (s)",
            "tracking_time_s": "Tracking time (s)",
            "num_corners": "Number of path corners",
            "inference_time_s": "Compute time (s)",
            "max_corner_deg": "Max corner degree (掳)",
            "planning_cost": "Planning cost (m)",
            "composite_score": "Composite score",
        }
    )
    table_mean_pretty.to_csv(out_dir / "table2_kpis_mean.csv", index=False)
    table_mean_pretty.to_markdown(out_dir / "table2_kpis_mean.md", index=False)

    # ---- 全部成功后过滤 ----
    if bool(getattr(args, "filter_all_succeed", False)) and not table.empty:
        # 对每个 (Environment, run_idx)，检查是否所有算法都到达了目标。
        _sr = table[["Environment", "Algorithm", "run_idx", "success_rate"]].copy()
        _sr["_ok"] = pd.to_numeric(_sr["success_rate"], errors="coerce").astype(float) >= 1.0 - 1e-9
        _all_ok = _sr.groupby(["Environment", "run_idx"], sort=False)["_ok"].all()
        _keep_all = sorted(_all_ok[_all_ok].index.tolist(), key=lambda t: int(t[1]))  # 按 run_idx 排序
        n_total = int(table.groupby("Environment", sort=False)["run_idx"].nunique().max()) if not table.empty else 0
        n_kept_raw = len({ri for _, ri in _keep_all})

        # --filter-target-count：截断为前 N 个全部成功的对。
        _ftc = int(getattr(args, "filter_target_count", 0))
        if _ftc > 0 and len(_keep_all) > _ftc:
            _keep_all = _keep_all[:_ftc]
            print(f"[filter-all-succeed] Truncated from {n_kept_raw} to {_ftc} pairs (--filter-target-count).")
        elif _ftc > 0 and len(_keep_all) < _ftc:
            print(
                f"[filter-all-succeed] WARNING: only {len(_keep_all)} all-succeed pairs found, "
                f"fewer than requested {_ftc}. Consider increasing --runs."
            )

        _keep = set(_keep_all)
        n_kept = len({ri for _, ri in _keep})
        print(f"[filter-all-succeed] Kept {n_kept}/{n_total} run pairs where all algorithms succeeded.")

        # 将全部成功的对保存到 JSON，供后续 --load-pairs 复用。
        _allsuc_pairs: list[dict[str, list[int]]] = []
        for _env_key, _ri in _keep_all:
            _pair_rows = table[(table["Environment"] == _env_key) & (table["run_idx"] == _ri)]
            if not _pair_rows.empty:
                _row0 = _pair_rows.iloc[0]
                _allsuc_pairs.append({
                    "start_xy": [int(_row0["start_x"]), int(_row0["start_y"])],
                    "goal_xy": [int(_row0["goal_x"]), int(_row0["goal_y"])],
                    "run_idx": int(_ri),
                })
        _pairs_out = out_dir / "allsuc_pairs.json"
        _pairs_out.write_text(
            json.dumps({"n_pairs": len(_allsuc_pairs), "pairs": _allsuc_pairs}, indent=2),
            encoding="utf-8",
        )
        print(f"[filter-all-succeed] Saved {len(_allsuc_pairs)} all-succeed pairs to {_pairs_out}")

        table_f = table[table.apply(lambda r: (str(r["Environment"]), int(r["run_idx"])) in _keep, axis=1)].copy()
        if table_f.empty:
            print("[filter-all-succeed] WARNING: no pairs survived the filter — skipping filtered tables.")
        else:
            # 从过滤后的原始行重新计算均值 KPI。
            _kpi_cols = [
                "avg_path_length", "path_time_s", "avg_curvature_1_m",
                "planning_time_s", "tracking_time_s", "num_corners",
                "inference_time_s", "max_corner_deg",
            ]
            _group = table_f.groupby(["Environment", "Algorithm"], sort=False)
            _n_runs_f = _group["run_idx"].nunique()
            _sr_f = _group["success_rate"].mean()
            _means = _group[_kpi_cols].mean()
            table_mean_f = _means.copy()
            table_mean_f["success_rate"] = _sr_f
            table_mean_f["n_filtered_runs"] = _n_runs_f
            table_mean_f = table_mean_f.reset_index()

            # 在过滤后的均值上重新计算综合指标。
            _w_t = float(args.score_time_weight)
            _sr_raw_f = pd.to_numeric(table_mean_f["success_rate"], errors="coerce").astype(float)
            _denom_f = _sr_raw_f.clip(lower=1e-6)
            _base_f = pd.to_numeric(table_mean_f["avg_path_length"], errors="coerce").astype(float) + _w_t * pd.to_numeric(
                table_mean_f["inference_time_s"], errors="coerce"
            ).astype(float)
            _pc_f = (_base_f / _denom_f).astype(float)
            _pc_f = _pc_f.where((_sr_raw_f > 0.0) & np.isfinite(_base_f.to_numpy()), other=float("inf"))
            table_mean_f["planning_cost"] = _pc_f

            _n_pt_f = table_mean_f.groupby(["Environment"], sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
            _n_k_f = table_mean_f.groupby(["Environment"], sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
            _n_pl_f = table_mean_f.groupby(["Environment"], sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
            _bs_f = (w_pt * _n_pt_f + w_k * _n_k_f + w_pl * _n_pl_f) / w_sum
            _sr_d2_f = _sr_raw_f.clip(lower=1e-6)
            _cs_f = (_bs_f / _sr_d2_f).astype(float)
            _cs_f = _cs_f.where(_sr_raw_f > 0.0, other=float("inf"))
            table_mean_f["composite_score"] = _cs_f

            # 四舍五入并写入。
            for c, r in [("success_rate", 3), ("avg_path_length", 4), ("path_time_s", 4),
                         ("avg_curvature_1_m", 6), ("planning_time_s", 5), ("tracking_time_s", 5),
                         ("inference_time_s", 5), ("planning_cost", 3), ("composite_score", 3)]:
                if c in table_mean_f.columns:
                    table_mean_f[c] = pd.to_numeric(table_mean_f[c], errors="coerce").astype(float).round(r)
            for c in ["num_corners", "max_corner_deg"]:
                if c in table_mean_f.columns:
                    table_mean_f[c] = pd.to_numeric(table_mean_f[c], errors="coerce").round(0).astype("Int64")

            # 写入过滤后的原始表。
            table_f.to_csv(out_dir / "table2_kpis_raw_filtered.csv", index=False)

            # 写入过滤后的均值表。
            table_mean_f_pretty = table_mean_f.rename(
                columns={
                    "Algorithm": "Algorithm name",
                    "success_rate": "Success rate",
                    "avg_path_length": "Average path length (m)",
                    "path_time_s": "Path time (s)",
                    "avg_curvature_1_m": "Average curvature (1/m)",
                    "planning_time_s": "Planning time (s)",
                    "tracking_time_s": "Tracking time (s)",
                    "num_corners": "Number of path corners",
                    "inference_time_s": "Compute time (s)",
                    "max_corner_deg": "Max corner degree (\u00b0)",
                    "planning_cost": "Planning cost (m)",
                    "composite_score": "Composite score",
                    "n_filtered_runs": "Filtered runs",
                }
            )
            table_mean_f_pretty.to_csv(out_dir / "table2_kpis_mean_filtered.csv", index=False)
            table_mean_f_pretty.to_markdown(out_dir / "table2_kpis_mean_filtered.md", index=False)
            print(f"Wrote: {out_dir / 'table2_kpis_raw_filtered.csv'}")
            print(f"Wrote: {out_dir / 'table2_kpis_mean_filtered.csv'}")
            print(f"Wrote: {out_dir / 'table2_kpis_mean_filtered.md'}")

    print(f"Wrote: {out_dir / 'table2_kpis.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis.md'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean.md'}")
    if bool(getattr(args, "filter_all_succeed", False)):
        for _fn in ["table2_kpis_raw_filtered.csv", "table2_kpis_mean_filtered.csv", "table2_kpis_mean_filtered.md"]:
            if (out_dir / _fn).exists():
                print(f"Wrote: {out_dir / _fn}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
