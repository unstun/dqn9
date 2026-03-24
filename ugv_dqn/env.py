"""UGV 路径规划 Gymnasium 环境模块。

本文件包含以下三个部分：

Section 1 — 工具函数
    _downsample_map_preserve_aspect   保持宽高比的占据栅格降采样。

Section 2 — 自行车模型原语（运动学、碰撞检测、代价场）
    BicycleModelParams                阿克曼自行车运动学参数。
    build_ackermann_action_table_35   35 离散动作表（转向角速率 × 加速度）。
    bicycle_integrate_one_step        单步欧拉积分（自行车 ODE）。
    TwoCircleFootprint                双圆近似碰撞足迹。
    compute_edt_distance_m            欧氏距离变换（EDT），用于安全距离场。
    bilinear_sample_2d*               亚像素双线性插值采样。
    dijkstra_cost_to_goal_m           Dijkstra 绕障最短路径距离场（goal distance field）。

Section 3 — UGVBicycleEnv（完整阿克曼自行车环境）
    UGVBicycleEnv                     基于自行车运动学的完整环境，包含：
        - 阿克曼转向 + 加速度（35 离散动作）
        - 三通道地图观测：占据栅格 + goal distance field + EDT 安全距离
        - 多分量奖励（目标接近、碰撞、进度塑形、安全距离、速度、曲率）
        - 可容许动作掩码 / 安全屏障
        - Hybrid A* 专家动作（用于 DQfD 演示）
        - 短视野启发式回退策略
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from ugv_dqn.maps import MapSpec


# ===========================================================================
# Section 1 — 工具函数
# ===========================================================================

def _downsample_map_preserve_aspect(
    src: np.ndarray,
    target_size: int,
    *,
    interpolation: int = cv2.INTER_AREA,
    pad_value: float = 0.0,
) -> np.ndarray:
    """保持宽高比地将 2D 地图降采样至 (target_size, target_size)。

    长边缩放至 target_size，短边等比例缩放后在底部填充 pad_value，
    保证输出始终为正方形。对于正方形输入，等价于直接 cv2.resize。
    """
    h, w = src.shape[:2]
    n = int(target_size)
    long_side = max(h, w)
    scale = float(n) / float(long_side)
    ds_w = max(1, round(w * scale))
    ds_h = max(1, round(h * scale))
    # 四舍五入可能超出 1 像素，需钳位到 target_size。
    ds_w = min(ds_w, n)
    ds_h = min(ds_h, n)
    resized = cv2.resize(
        src.astype(np.float32, copy=False),
        dsize=(int(ds_w), int(ds_h)),
        interpolation=interpolation,
    )
    if ds_h == n and ds_w == n:
        return resized.astype(np.float32, copy=False)
    out = np.full((n, n), float(pad_value), dtype=np.float32)
    out[:ds_h, :ds_w] = resized
    return out


# ===========================================================================
# Section 2 — 自行车模型原语（运动学、碰撞检测、代价场）
# ===========================================================================

@dataclass(frozen=True)
class BicycleModelParams:
    dt: float = 0.05
    wheelbase_m: float = 0.6

    v_max_m_s: float = 2.0
    a_max_m_s2: float = 1.5

    delta_max_rad: float = math.radians(27.0)
    omega_max_rad_s: float = 1.223
    delta_dot_max_rad_s: float = math.radians(60.0)


def build_ackermann_action_table_35(*, delta_dot_max_rad_s: float, a_max_m_s2: float) -> np.ndarray:
    """返回 (35, 2) 动作表，列为 [转向角速率 delta_dot(rad/s), 加速度 a(m/s²)]。"""
    dd = float(delta_dot_max_rad_s)
    aa = float(a_max_m_s2)
    delta_dots = np.array(
        [-dd, -(2.0 / 3.0) * dd, -(1.0 / 3.0) * dd, 0.0, (1.0 / 3.0) * dd, (2.0 / 3.0) * dd, dd],
        dtype=np.float32,
    )
    accels = np.array([-aa, -0.5 * aa, 0.0, 0.5 * aa, aa], dtype=np.float32)
    table = np.zeros((delta_dots.size * accels.size, 2), dtype=np.float32)
    k = 0
    for d_dot in delta_dots:
        for a in accels:
            table[k, 0] = float(d_dot)
            table[k, 1] = float(a)
            k += 1
    return table


def wrap_angle_rad(x: float) -> float:
    """将角度归一化到 [-π, π) 范围。"""
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def bicycle_integrate_one_step(
    *,
    x_m: float,
    y_m: float,
    psi_rad: float,
    v_m_s: float,
    delta_rad: float,
    delta_dot_rad_s: float,
    a_m_s2: float,
    params: BicycleModelParams,
) -> tuple[float, float, float, float, float]:
    """后轴中心自行车模型，单步欧拉积分。"""
    dt = float(params.dt)
    v_next = float(np.clip(v_m_s + float(a_m_s2) * dt, -float(params.v_max_m_s), float(params.v_max_m_s)))

    delta_unclipped = float(delta_rad) + float(delta_dot_rad_s) * dt
    delta_lim = float(params.delta_max_rad)
    delta_next = float(np.clip(delta_unclipped, -float(delta_lim), +float(delta_lim)))

    x_next = float(x_m) + v_next * math.cos(float(psi_rad)) * dt
    y_next = float(y_m) + v_next * math.sin(float(psi_rad)) * dt
    psi_next = wrap_angle_rad(float(psi_rad) + (v_next / float(params.wheelbase_m)) * math.tan(delta_next) * dt)
    return x_next, y_next, psi_next, v_next, delta_next


def min_steps_to_cover_distance_m(
    distance_m: float,
    *,
    dt: float,
    v_max_m_s: float,
    a_max_m_s2: float,
    v0_m_s: float = 0.0,
) -> int:
    """沿直线覆盖 distance_m 所需的最少步数。

    使用与环境速度积分器相同的离散更新规则：
        v_{k+1} = clip(v_k + a_max * dt, 0, v_max)
        x_{k+1} = x_k + v_{k+1} * dt
    """
    dist = max(0.0, float(distance_m))
    if dist <= 0.0:
        return 0

    dt_ = float(dt)
    if not (dt_ > 0.0):
        raise ValueError("dt must be > 0")
    v_max = float(v_max_m_s)
    if not (v_max > 0.0):
        raise ValueError("v_max_m_s must be > 0")
    a_max = float(a_max_m_s2)
    if not (a_max > 0.0):
        raise ValueError("a_max_m_s2 must be > 0")

    v = max(0.0, float(v0_m_s))
    covered = 0.0
    steps = 0
    while covered < dist:
        steps += 1
        v = min(v_max, v + a_max * dt_)
        covered += v * dt_
        if steps > 1_000_000:
            raise RuntimeError("min_steps_to_cover_distance_m exceeded step limit; check inputs.")
    return int(steps)


@dataclass(frozen=True)
class TwoCircleFootprint:
    radius_m: float = float(math.hypot(0.740 / 2.0, 0.924 / 4.0))
    x1_m: float = float((0.6 / 2.0) - (0.924 / 4.0))
    x2_m: float = float((0.6 / 2.0) + (0.924 / 4.0))


def compute_edt_distance_m(grid_y0_bottom: np.ndarray, *, cell_size_m: float) -> np.ndarray:
    """EDT 距离（米），每个栅格中心到最近障碍物栅格中心的欧氏距离。"""
    grid_top = grid_y0_bottom[::-1, :]
    free = (grid_top == 0).astype(np.uint8) * 255
    dist_top = cv2.distanceTransform(
        free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    return (dist_top[::-1, :] * float(cell_size_m)).astype(np.float32, copy=False)


def bilinear_sample_2d(arr: np.ndarray, *, x: float, y: float, default: float = float("inf")) -> float:
    """在 (H, W) 数组上按索引坐标 (x, y) 进行双线性插值采样。"""
    h, w = arr.shape
    if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
        return float(default)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x - x0)
    fy = float(y - y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def bilinear_sample_2d_finite(
    arr: np.ndarray,
    *,
    x: float,
    y: float,
    fill_value: float,
    default: float | None = None,
) -> float:
    """将非有限角点值替换为 fill_value 后再双线性插值。

    用于采样 goal distance field 时非常重要：该场用 inf 标记不可通行栅格，
    普通双线性插值会把 inf 传播到相邻有效区域，破坏障碍物附近的塑形梯度。
    """

    h, w = arr.shape
    if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
        return float(fill_value if default is None else default)

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x - x0)
    fy = float(y - y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])

    fv = float(fill_value)
    if not math.isfinite(v00):
        v00 = fv
    if not math.isfinite(v10):
        v10 = fv
    if not math.isfinite(v01):
        v01 = fv
    if not math.isfinite(v11):
        v11 = fv

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def bilinear_sample_2d_vec(
    arr: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    default: float = float("inf"),
) -> np.ndarray:
    """向量化双线性插值采样，输入为索引坐标 (x, y) 数组。"""
    h, w = arr.shape
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    if h == 0 or w == 0:
        return np.full_like(xv, float(default), dtype=np.float64)

    mask = (xv >= 0.0) & (xv <= float(w - 1)) & (yv >= 0.0) & (yv <= float(h - 1))
    x_c = np.clip(xv, 0.0, float(w - 1))
    y_c = np.clip(yv, 0.0, float(h - 1))

    x0 = np.floor(x_c).astype(np.int32, copy=False)
    y0 = np.floor(y_c).astype(np.int32, copy=False)
    x1 = np.minimum(x0 + 1, int(w - 1)).astype(np.int32, copy=False)
    y1 = np.minimum(y0 + 1, int(h - 1)).astype(np.int32, copy=False)

    fx = x_c - x0.astype(np.float64, copy=False)
    fy = y_c - y0.astype(np.float64, copy=False)

    v00 = arr[y0, x0].astype(np.float64, copy=False)
    v10 = arr[y0, x1].astype(np.float64, copy=False)
    v01 = arr[y1, x0].astype(np.float64, copy=False)
    v11 = arr[y1, x1].astype(np.float64, copy=False)

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    out = v0 * (1.0 - fy) + v1 * fy
    return np.where(mask, out, float(default)).astype(np.float64, copy=False)


def bilinear_sample_2d_finite_vec(
    arr: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float,
    default: float | None = None,
) -> np.ndarray:
    """向量化双线性插值，非有限角点值替换为 fill_value（用于 goal distance field 采样）。"""
    h, w = arr.shape
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    if h == 0 or w == 0:
        fv = float(fill_value if default is None else default)
        return np.full_like(xv, fv, dtype=np.float64)

    mask = (xv >= 0.0) & (xv <= float(w - 1)) & (yv >= 0.0) & (yv <= float(h - 1))
    x_c = np.clip(xv, 0.0, float(w - 1))
    y_c = np.clip(yv, 0.0, float(h - 1))

    x0 = np.floor(x_c).astype(np.int32, copy=False)
    y0 = np.floor(y_c).astype(np.int32, copy=False)
    x1 = np.minimum(x0 + 1, int(w - 1)).astype(np.int32, copy=False)
    y1 = np.minimum(y0 + 1, int(h - 1)).astype(np.int32, copy=False)

    fx = x_c - x0.astype(np.float64, copy=False)
    fy = y_c - y0.astype(np.float64, copy=False)

    v00 = arr[y0, x0].astype(np.float64, copy=False)
    v10 = arr[y0, x1].astype(np.float64, copy=False)
    v01 = arr[y1, x0].astype(np.float64, copy=False)
    v11 = arr[y1, x1].astype(np.float64, copy=False)

    fv = float(fill_value)
    v00 = np.where(np.isfinite(v00), v00, fv)
    v10 = np.where(np.isfinite(v10), v10, fv)
    v01 = np.where(np.isfinite(v01), v01, fv)
    v11 = np.where(np.isfinite(v11), v11, fv)

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    out = v0 * (1.0 - fy) + v1 * fy

    outside = float(fill_value if default is None else default)
    return np.where(mask, out, outside).astype(np.float64, copy=False)


def dijkstra_cost_to_goal_m(
    traversable_y0_bottom: np.ndarray,
    *,
    goal_xy: tuple[int, int],
    cell_size_m: float,
) -> np.ndarray:
    """通过 Dijkstra 算法计算 8 连通绕障最短路径距离场（goal distance field，单位：米）。"""
    if traversable_y0_bottom.ndim != 2:
        raise ValueError("traversable_y0_bottom must be a 2D array")
    h, w = traversable_y0_bottom.shape
    if h == 0 or w == 0:
        raise ValueError("traversable_y0_bottom must be non-empty")
    cell = float(cell_size_m)
    if not (cell > 0.0):
        raise ValueError("cell_size_m must be > 0")

    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if not (0 <= gx < w and 0 <= gy < h):
        raise ValueError("goal_xy is out of bounds")

    traversable = traversable_y0_bottom.astype(bool, copy=False)

    cost = np.full((h, w), float("inf"), dtype=np.float64)
    if not bool(traversable[gy, gx]):
        return cost

    pq: list[tuple[float, int, int]] = []
    cost[gy, gx] = 0.0
    heapq.heappush(pq, (0.0, gx, gy))

    moves: tuple[tuple[int, int, float], ...] = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    )

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != float(cost[y, x]):
            continue
        for dx, dy, step in moves:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if not bool(traversable[ny, nx]):
                continue
            nd = float(d) + float(step) * cell
            if nd < float(cost[ny, nx]):
                cost[ny, nx] = float(nd)
                heapq.heappush(pq, (float(nd), nx, ny))

    return cost.astype(np.float32, copy=False)


# ===========================================================================
# Section 3 — UGVBicycleEnv（完整阿克曼自行车运动学环境）
# ===========================================================================

class UGVBicycleEnv(gym.Env):
    """基于阿克曼/自行车运动学的栅格环境，使用 EDT 进行碰撞检测和安全距离（OD）计算。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_spec: MapSpec,
        *,
        max_steps: int = 500,
        cell_size_m: float = 0.1,
        model: BicycleModelParams = BicycleModelParams(),
        footprint: TwoCircleFootprint = TwoCircleFootprint(),
        sensor_range_m: float = 6.0,
        n_sectors: int = 36,
        obs_map_size: int = 12,
        od_cap_m: float = 2.0,
        safe_distance_m: float = 0.20,
        safe_speed_distance_m: float = 0.20,
        # 0.3 m 与 DRL 导航文献主流设置一致（Cimurs et al.; ROS 2 Nav2 默认 0.25 m）
        goal_tolerance_m: float = 0.3,
        goal_angle_tolerance_deg: float = 180.0,
        goal_speed_tol_m_s: float = 999.0,
        reward_k_p: float = 12.0,
        reward_k_t: float = 0.2,
        reward_k_delta: float = 1.5,
        reward_k_a: float = 0.2,
        reward_k_kappa: float = 0.2,
        reward_k_o: float = 1.5,
        reward_k_v: float = 2.0,
        reward_k_c: float = 0.0,
        reward_k_goal: float = 0.0,
        reward_k_eff: float = 0.0,
        reward_obs_max: float = 10.0,
        stuck_steps: int = 20,
        stuck_min_disp_m: float = 0.02,
        stuck_min_speed_m_s: float = 0.05,
        stuck_penalty: float = 300.0,
        # 消融实验表明 "diag" 模式（碰撞边距 = √2/2 * cell_size，即对角线半栅格）
        # 比 "half" 模式（碰撞边距 = 0.5 * cell_size）更保守，碰撞率更低，
        # 尤其在狭窄走廊场景下显著提升成功率。默认使用 "diag"。
        edt_collision_margin: str = "diag",
        scalar_only: bool = False,
    ) -> None:
        super().__init__()

        self.map_spec = map_spec
        self._grid = map_spec.obstacle_grid().astype(np.uint8, copy=False)  # (H, W)，y=0 在底部
        self._height, self._width = self._grid.shape
        self._canonical_start_xy = (int(map_spec.start_xy[0]), int(map_spec.start_xy[1]))
        self._canonical_goal_xy = (int(map_spec.goal_xy[0]), int(map_spec.goal_xy[1]))
        self.start_xy = (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
        self.goal_xy = (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))

        self.max_steps = int(max_steps)
        self.cell_size_m = float(cell_size_m)
        if not (self.cell_size_m > 0):
            raise ValueError("cell_size_m must be > 0")

        self.model = model
        self.footprint = footprint
        self.sensor_range_m = float(sensor_range_m)
        if not (self.sensor_range_m > 0):
            raise ValueError("sensor_range_m must be > 0")
        self.n_sectors = int(n_sectors)
        if self.n_sectors < 1:
            raise ValueError("n_sectors must be >= 1")
        self.scalar_only = bool(scalar_only)
        self.obs_map_size = int(obs_map_size)
        if not self.scalar_only and self.obs_map_size < 4:
            raise ValueError("obs_map_size must be >= 4")
        self.od_cap_m = float(od_cap_m)
        if not (self.od_cap_m > 0):
            raise ValueError("od_cap_m must be > 0")
        self.safe_distance_m = float(safe_distance_m)
        if not (self.safe_distance_m > 0):
            raise ValueError("safe_distance_m must be > 0")
        self.safe_speed_distance_m = float(safe_speed_distance_m)
        if not (self.safe_speed_distance_m > 0):
            raise ValueError("safe_speed_distance_m must be > 0")
        if float(self.safe_speed_distance_m) < float(self.safe_distance_m):
            raise ValueError("safe_speed_distance_m must be >= safe_distance_m")
        self.goal_tolerance_m = float(goal_tolerance_m)
        if not (self.goal_tolerance_m > 0):
            raise ValueError("goal_tolerance_m must be > 0")
        self.goal_angle_tolerance_rad = float(math.radians(float(goal_angle_tolerance_deg)))
        if not (0.0 < self.goal_angle_tolerance_rad <= math.pi):
            raise ValueError("goal_angle_tolerance_deg must be in (0, 180]")
        self.goal_speed_tol_m_s = float(goal_speed_tol_m_s)

        # 预计算 EDT 和 goal distance field（森林地图为静态地图，只需计算一次）。
        self._eps_cell_m = float(math.sqrt(2.0) * 0.5 * self.cell_size_m)
        # EDT 碰撞边距：
        #   "diag" → √2/2 * cell_size（对角线半栅格，更保守，消融实验推荐）
        #   "half" → 0.5 * cell_size（正交半栅格，旧默认值）
        if edt_collision_margin == "diag":
            self._half_cell_m = float(math.sqrt(2.0) * 0.5 * self.cell_size_m)
        else:
            self._half_cell_m = float(0.5 * self.cell_size_m)
        self._dist_m = compute_edt_distance_m(self._grid, cell_size_m=self.cell_size_m)
        self._diag_m = float(
            math.hypot(float(self._width - 1) * self.cell_size_m, float(self._height - 1) * self.cell_size_m)
        )

        # 将世界边界也视为障碍物，同时用于碰撞检测和传感。
        max_x = float(self._width - 1) * self.cell_size_m
        max_y = float(self._height - 1) * self.cell_size_m
        xs = (np.arange(self._width, dtype=np.float32) * float(self.cell_size_m)).reshape(1, -1)
        ys = (np.arange(self._height, dtype=np.float32) * float(self.cell_size_m)).reshape(-1, 1)
        boundary_dist = np.minimum(
            np.minimum(xs, float(max_x) - xs),
            np.minimum(ys, float(max_y) - ys),
        ).astype(np.float32, copy=False)
        self._dist_m = np.minimum(self._dist_m, boundary_dist).astype(np.float32, copy=False)

        # 可通行性掩码，用于 goal distance field 塑形和课程学习采样。
        #
        # 使用无碰撞安全距离（r + eps_cell）。奖励函数中的 OD 安全距离项
        # 会额外处理边距；若代价场过于保守会断开自由空间，消除有用的进度梯度。
        self._clearance_thr_m = float(self.footprint.radius_m) + float(self._eps_cell_m)
        self._traversable_base = (self._dist_m > float(self._clearance_thr_m)).astype(bool, copy=False)
        # 确保固定起点/终点栅格始终被视为可通行。
        self._traversable_base[self._canonical_start_xy[1], self._canonical_start_xy[0]] = True
        self._traversable_base[self._canonical_goal_xy[1], self._canonical_goal_xy[0]] = True

        # 用于随机起点/终点采样的候选自由栅格。
        free_y, free_x = np.where(self._traversable_base)
        self._rand_free_xy = np.stack([free_x, free_y], axis=1).astype(np.int32, copy=False)

        # 目标相关的字段（goal distance field + 课程学习候选点）。
        self._set_goal_xy(self.goal_xy)
        # 起点相关的归一化 + 降采样代价图。
        self._update_start_dependent_fields(start_xy=self.start_xy)

        # 合理性检查：基于起点的 goal distance（考虑绕障）验证步数上限是否充足。
        min_steps = min_steps_to_cover_distance_m(
            max(0.0, float(self._cost_norm_m) - float(self.goal_tolerance_m)),
            dt=float(self.model.dt),
            v_max_m_s=float(self.model.v_max_m_s),
            a_max_m_s2=float(self.model.a_max_m_s2),
            v0_m_s=0.0,
        )
        if int(self.max_steps) < int(min_steps):
            raise ValueError(
                f"max_steps={self.max_steps} is too small for forest env {self.map_spec.name!r} "
                f"(cost-to-go≈{self._cost_norm_m:.2f}m with v_max={self.model.v_max_m_s:.2f}m/s, "
                f"a_max={self.model.a_max_m_s2:.2f}m/s², dt={self.model.dt:.3f}s). "
                f"Need at least {min_steps} steps (increase --max-steps)."
            )

        self.reward_k_p = float(reward_k_p)
        self.reward_k_t = float(reward_k_t)
        self.reward_k_delta = float(reward_k_delta)
        self.reward_k_a = float(reward_k_a)
        self.reward_k_kappa = float(reward_k_kappa)
        self.reward_k_o = float(reward_k_o)
        self.reward_k_v = float(reward_k_v)
        self.reward_k_c = float(reward_k_c)
        self.reward_k_goal = float(reward_k_goal)
        self.reward_k_eff = float(reward_k_eff)
        self.reward_obs_max = float(reward_obs_max)
        self.reward_eps = 1e-3
        self.stuck_steps = int(stuck_steps)
        if self.stuck_steps < 1:
            raise ValueError("stuck_steps must be >= 1")
        self.stuck_min_disp_m = float(stuck_min_disp_m)
        if not (self.stuck_min_disp_m >= 0.0):
            raise ValueError("stuck_min_disp_m must be >= 0")
        self.stuck_min_speed_m_s = float(stuck_min_speed_m_s)
        if not (self.stuck_min_speed_m_s >= 0.0):
            raise ValueError("stuck_min_speed_m_s must be >= 0")
        self.stuck_penalty = float(stuck_penalty)
        if not (self.stuck_penalty >= 0.0):
            raise ValueError("stuck_penalty must be >= 0")

        self.action_table = build_ackermann_action_table_35(
            delta_dot_max_rad_s=float(model.delta_dot_max_rad_s),
            a_max_m_s2=float(model.a_max_m_s2),
        )
        self.action_space = gym.spaces.Discrete(int(self.action_table.shape[0]))

        # 全局规划观测：智能体/目标位姿 + 降采样静态地图。
        #
        # 障碍栅格和 goal distance field 在全局规划中已知，比纯激光雷达特征
        # 提供更丰富的上下文信息。
        #
        # 保持宽高比降采样，使非正方形地图（如 410×129）保持正确的空间关系。
        # 填充值语义正确：占据=1，最大代价=1，零安全距离=0。
        if self.scalar_only:
            # 消融实验：仅保留 11 维标量特征，去除全部地图通道。
            self._obs_occ_flat = np.array([], dtype=np.float32)
            self._obs_cost_flat = np.array([], dtype=np.float32)
            self._obs_edt_flat = np.array([], dtype=np.float32)
            obs_dim = 11
        else:
            _n = int(self.obs_map_size)
            occ_ds = _downsample_map_preserve_aspect(
                self._grid.astype(np.float32, copy=False), _n,
                interpolation=cv2.INTER_NEAREST, pad_value=1.0,
            )
            self._obs_occ_flat = (2.0 * occ_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

            cost = np.minimum(self._cost_to_goal_m, float(self._cost_fill_m)).astype(np.float32, copy=False)
            cost01 = np.clip(cost / max(1e-6, float(self._cost_norm_m)), 0.0, 1.0).astype(np.float32, copy=False)
            cost_ds = _downsample_map_preserve_aspect(cost01, _n, pad_value=1.0)
            self._obs_cost_flat = (2.0 * cost_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

            # EDT 安全距离图：归一化到最近障碍物的距离，上限为 od_cap_m。
            edt01 = np.clip(self._dist_m / max(1e-6, float(self.od_cap_m)), 0.0, 1.0).astype(np.float32, copy=False)
            edt_ds = _downsample_map_preserve_aspect(edt01, _n, pad_value=0.0)
            self._obs_edt_flat = (2.0 * edt_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

            obs_dim = 11 + 3 * int(self.obs_map_size) * int(self.obs_map_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._steps = 0

        self._x_m = float(self.start_xy[0]) * self.cell_size_m
        self._y_m = float(self.start_xy[1]) * self.cell_size_m
        self._psi_rad = 0.0
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._prev_delta_dot = 0.0
        self._prev_a = 0.0
        self._last_od_m = 0.0
        self._last_collision = False
        self._stuck_pos_history: list[tuple[float, float]] = []
        self._ha_path_cache: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}
        self._ha_progress_idx: int = 0
        self._ha_start_xy: tuple[int, int] = self.start_xy

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def _in_bounds_xy(self, xy: tuple[int, int]) -> bool:
        x, y = int(xy[0]), int(xy[1])
        return 0 <= x < int(self._width) and 0 <= y < int(self._height)

    def _set_goal_xy(self, goal_xy: tuple[int, int]) -> None:
        gx, gy = int(goal_xy[0]), int(goal_xy[1])
        if not self._in_bounds_xy((gx, gy)):
            raise ValueError("goal_xy is out of bounds")

        self.goal_xy = (int(gx), int(gy))

        traversable = self._traversable_base
        if not bool(traversable[int(gy), int(gx)]):
            trav = traversable.astype(bool, copy=True)
            trav[int(gy), int(gx)] = True
        else:
            trav = traversable

        self._cost_to_goal_m = dijkstra_cost_to_goal_m(
            trav,
            goal_xy=self.goal_xy,
            cell_size_m=self.cell_size_m,
        )
        finite_cost = self._cost_to_goal_m[np.isfinite(self._cost_to_goal_m)]
        if finite_cost.size == 0:
            raise ValueError(
                f"Forest map {self.map_spec.name!r} has no reachable states for goal={self.goal_xy}; "
                "pick a different goal or regenerate the map."
            )
        self._cost_fill_m = float(np.max(finite_cost)) + float(self.cell_size_m)

        # 课程学习：候选起始栅格（安全距离可通行 + 有限 goal distance）。
        self._curriculum_min_cost_m = float(max(self.goal_tolerance_m + self.cell_size_m, 1.0))
        cand_mask = np.isfinite(self._cost_to_goal_m) & (self._dist_m > float(self._clearance_thr_m))
        # 排除目标栅格（过于简单）及目标容差范围内的栅格。
        cand_mask[int(gy), int(gx)] = False
        cand_mask &= self._cost_to_goal_m >= float(self._curriculum_min_cost_m)

        cand_y, cand_x = np.where(cand_mask)
        self._curriculum_start_xy = np.stack([cand_x, cand_y], axis=1).astype(np.int32, copy=False)
        self._curriculum_start_costs_m = self._cost_to_goal_m[cand_y, cand_x].astype(np.float32, copy=False)

    def _heading_from_cost_gradient(self, cx: int, cy: int) -> float | None:
        """根据 goal distance field 梯度下降方向返回航向角，无法计算时返回 None。"""
        cost = self._cost_to_goal_m
        h, w = cost.shape
        x0, x1 = max(0, cx - 1), min(w - 1, cx + 1)
        y0, y1 = max(0, cy - 1), min(h - 1, cy + 1)
        c_x0, c_x1 = float(cost[cy, x0]), float(cost[cy, x1])
        c_y0, c_y1 = float(cost[y0, cx]), float(cost[y1, cx])
        if not (math.isfinite(c_x0) and math.isfinite(c_x1)
                and math.isfinite(c_y0) and math.isfinite(c_y1)):
            return None
        gx = c_x1 - c_x0
        gy = c_y1 - c_y0
        if abs(gx) < 1e-12 and abs(gy) < 1e-12:
            return None
        return wrap_angle_rad(math.atan2(-gy, -gx))

    def _update_start_dependent_fields(self, *, start_xy: tuple[int, int]) -> None:
        sx, sy = int(start_xy[0]), int(start_xy[1])
        if not self._in_bounds_xy((sx, sy)):
            raise ValueError("start_xy is out of bounds")

        # 用锚定起始位姿（双圆足迹）归一化 goal distance，而非仅用后轴栅格，
        # 使得障碍物附近的塑形保持有效。
        start_cost_cell = float(self._cost_to_goal_m[int(sy), int(sx)])
        start_x_m = float(sx) * self.cell_size_m
        start_y_m = float(sy) * self.cell_size_m
        dx0 = float(self.goal_xy[0] - int(sx)) * self.cell_size_m
        dy0 = float(self.goal_xy[1] - int(sy)) * self.cell_size_m
        psi0 = wrap_angle_rad(math.atan2(dy0, dx0))
        _od_chk, coll_chk = self._od_and_collision_at_pose_m(start_x_m, start_y_m, float(psi0))
        if bool(coll_chk):
            psi_grad = self._heading_from_cost_gradient(int(sx), int(sy))
            if psi_grad is not None:
                psi0 = psi_grad
        start_cost_pose = self._cost_to_goal_pose_m(start_x_m, start_y_m, psi0)
        self._cost_norm_m = float(max(self._diag_m, start_cost_cell, start_cost_pose))
        if not math.isfinite(self._cost_norm_m):
            raise ValueError(
                f"Forest map {self.map_spec.name!r} is unreachable under the clearance constraint; "
                "regenerate the map or reduce obstacle density."
            )

        # 课程学习代价锚点（仅用于按代价带采样起点）。
        start_cost = float(start_cost_pose if math.isfinite(start_cost_pose) else start_cost_cell)
        self._curriculum_start_cost_m = float(start_cost)

        # 降采样归一化 goal distance field，用于全局地图观测。
        if not self.scalar_only:
            cost = np.minimum(self._cost_to_goal_m, float(self._cost_fill_m)).astype(np.float32, copy=False)
            cost01 = np.clip(cost / max(1e-6, float(self._cost_norm_m)), 0.0, 1.0).astype(np.float32, copy=False)
            cost_ds = _downsample_map_preserve_aspect(
                cost01, int(self.obs_map_size), pad_value=1.0,
            )
            self._obs_cost_flat = (2.0 * cost_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

    def _sample_random_start_goal(
        self,
        *,
        min_cost_m: float,
        max_cost_m: float | None,
        fixed_prob: float,
        tries: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if self._rand_free_xy.size == 0:
            return self._canonical_start_xy, self._canonical_goal_xy

        p_fixed = float(np.clip(float(fixed_prob), 0.0, 1.0))
        if float(self._rng.random()) < p_fixed:
            self._set_goal_xy(self._canonical_goal_xy)
            self._update_start_dependent_fields(start_xy=self._canonical_start_xy)
            return self._canonical_start_xy, self._canonical_goal_xy

        min_cost = max(0.0, float(min_cost_m))
        max_cost = None if max_cost_m is None else max(0.0, float(max_cost_m))
        n_tries = max(1, int(tries))

        for _ in range(n_tries):
            gi = int(self._rng.integers(0, int(self._rand_free_xy.shape[0])))
            gx, gy = (int(self._rand_free_xy[gi, 0]), int(self._rand_free_xy[gi, 1]))

            try:
                self._set_goal_xy((gx, gy))
            except Exception:
                continue

            costs = self._cost_to_goal_m
            cand_mask = np.isfinite(costs) & (self._dist_m > float(self._clearance_thr_m))
            cand_mask[int(gy), int(gx)] = False
            if min_cost > 0.0:
                cand_mask &= costs >= float(min_cost)
            if max_cost is not None and max_cost > 0.0:
                cand_mask &= costs <= float(max_cost)

            sy, sx = np.where(cand_mask)
            if sx.size == 0:
                continue

            si = int(self._rng.integers(0, int(sx.size)))
            start_xy = (int(sx[si]), int(sy[si]))

            # 验证初始位姿在双圆足迹下无碰撞。
            # 先尝试 atan2 航向（朝目标方向），碰撞则回退到代价梯度航向。
            dx0 = float(gx - int(start_xy[0])) * float(self.cell_size_m)
            dy0 = float(gy - int(start_xy[1])) * float(self.cell_size_m)
            psi0 = wrap_angle_rad(math.atan2(dy0, dx0))
            sx_m = float(start_xy[0]) * float(self.cell_size_m)
            sy_m = float(start_xy[1]) * float(self.cell_size_m)
            _od0, coll0 = self._od_and_collision_at_pose_m(sx_m, sy_m, float(psi0))
            if bool(coll0):
                psi_grad = self._heading_from_cost_gradient(int(start_xy[0]), int(start_xy[1]))
                if psi_grad is None:
                    continue
                _od0, coll0 = self._od_and_collision_at_pose_m(sx_m, sy_m, float(psi_grad))
                if bool(coll0):
                    continue

            # 将归一化锚点更新为采样的起点。
            try:
                self._update_start_dependent_fields(start_xy=start_xy)
            except Exception:
                continue

            # 拒绝无法在回合步数上限内完成的起终点对。
            try:
                min_steps = min_steps_to_cover_distance_m(
                    max(0.0, float(self._cost_norm_m) - float(self.goal_tolerance_m)),
                    dt=float(self.model.dt),
                    v_max_m_s=float(self.model.v_max_m_s),
                    a_max_m_s2=float(self.model.a_max_m_s2),
                    v0_m_s=0.0,
                )
            except Exception:
                continue
            if int(self.max_steps) < int(min_steps):
                continue

            return start_xy, (gx, gy)

        # 回退：使用标准起终点对。
        self._set_goal_xy(self._canonical_goal_xy)
        self._update_start_dependent_fields(start_xy=self._canonical_start_xy)
        return self._canonical_start_xy, self._canonical_goal_xy

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0

        # 默认：使用固定的标准起点/终点（向后兼容）。
        start_xy = (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
        goal_xy = (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))

        start_override: tuple[int, int] | None = None
        goal_override: tuple[int, int] | None = None
        random_start_goal = False
        rand_min_cost_m = 0.0
        rand_max_cost_m: float | None = None
        rand_fixed_prob = 0.0
        rand_tries = 200

        if options:
            if options.get("start_xy") is not None:
                sx, sy = options["start_xy"]
                start_override = (int(sx), int(sy))
            if options.get("goal_xy") is not None:
                gx, gy = options["goal_xy"]
                goal_override = (int(gx), int(gy))
            random_start_goal = bool(options.get("random_start_goal", False))
            rand_min_cost_m = float(options.get("rand_min_cost_m", 0.0))
            max_raw = options.get("rand_max_cost_m", None)
            rand_max_cost_m = None if max_raw is None else float(max_raw)
            rand_fixed_prob = float(options.get("rand_fixed_prob", 0.0))
            rand_tries = int(options.get("rand_tries", 200))

        if goal_override is not None:
            goal_xy = (int(goal_override[0]), int(goal_override[1]))

        if random_start_goal and start_override is None and goal_override is None:
            start_xy, goal_xy = self._sample_random_start_goal(
                min_cost_m=float(rand_min_cost_m),
                max_cost_m=rand_max_cost_m,
                fixed_prob=float(rand_fixed_prob),
                tries=int(rand_tries),
            )
        else:
            # 确保环境的目标相关字段与请求的目标一致。
            if (int(self.goal_xy[0]), int(self.goal_xy[1])) != (int(goal_xy[0]), int(goal_xy[1])):
                self._set_goal_xy(goal_xy)

            if start_override is not None:
                start_xy = (int(start_override[0]), int(start_override[1]))
                # 显式指定 (start, goal) 时，以本次起点归一化 goal distance field。
                self._update_start_dependent_fields(start_xy=start_xy)
            else:
                # 归一化锚定在标准起点（与固定起点训练行为一致），
                # 即使课程学习采样了不同的起点也不改变。
                self._update_start_dependent_fields(start_xy=self._canonical_start_xy)

        self.start_xy = (int(start_xy[0]), int(start_xy[1]))
        self.goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

        ha_start_xy = (int(self.start_xy[0]), int(self.start_xy[1]))
        ha_progress_idx = 0
        psi_override: float | None = None
        if (
            options
            and (not bool(random_start_goal))
            and start_override is None
            and goal_override is None
            and options.get("curriculum_progress") is not None
            and len(self._curriculum_start_xy) > 0
        ):
            # 森林课程学习：早期回合从更靠近目标的位置出发；后期逐步将概率质量
            # 移回标准起点。防止训练/测试不匹配（训练时从未练习真实起点，推理时却总从那里开始）。
            p = float(options["curriculum_progress"])
            p = float(np.clip(p, 0.0, 1.0))

            # 以概率 p 使用固定起点（p=1 时始终从标准起点出发）。
            if float(self._rng.random()) >= float(p):
                band_m = float(options.get("curriculum_band_m", 2.0))
                band_m = max(float(self.cell_size_m), float(band_m))

                hi = float(self._curriculum_min_cost_m) + p * float(
                    max(0.0, float(self._curriculum_start_cost_m) - float(self._curriculum_min_cost_m))
                )
                lo = max(float(self._curriculum_min_cost_m), float(hi) - float(band_m))

                # 优先沿预计算的 Hybrid A* 参考路径采样起点。
                # 保持课程起点在已知可行走廊上，避免从大量随机起点重复规划（大森林地图会超时）。
                chosen_ref_idx: int | None = None
                ref_path = self._hybrid_astar_path(start_xy=self.start_xy)
                if len(ref_path) >= 2:
                    # 按参考路径进度采样（比将连续 Hybrid A* 坐标四舍五入回栅格后匹配
                    # 精确代价带更鲁棒）。
                    max_idx = max(0, int(len(ref_path) - 2))  # exclude last point (goal vicinity)
                    band_steps = max(1, int(round(float(band_m) / float(self.cell_size_m))))
                    target_idx = int(round((1.0 - float(p)) * float(max_idx)))
                    lo_i = max(0, int(target_idx) - int(band_steps))
                    hi_i = min(int(max_idx), int(target_idx) + int(band_steps))

                    ref_idxs: list[int] = []
                    for i in range(int(lo_i), int(hi_i) + 1):
                        px, py = ref_path[int(i)]
                        ix = int(round(float(px)))
                        iy = int(round(float(py)))
                        if not (0 <= ix < self._width and 0 <= iy < self._height):
                            continue
                        c = float(self._cost_to_goal_m[iy, ix])
                        if not math.isfinite(c) or float(c) < float(self._curriculum_min_cost_m):
                            continue
                        ref_idxs.append(int(i))

                    if ref_idxs:
                        chosen_ref_idx = int(self._rng.choice(ref_idxs))
                        px, py = ref_path[int(chosen_ref_idx)]
                        start_xy = (int(round(float(px))), int(round(float(py))))
                        ha_start_xy = (int(self.start_xy[0]), int(self.start_xy[1]))
                        ha_progress_idx = int(chosen_ref_idx)
                        j = min(int(chosen_ref_idx) + 1, len(ref_path) - 1)
                        px2, py2 = ref_path[int(j)]
                        dx = (float(px2) - float(px)) * float(self.cell_size_m)
                        dy = (float(py2) - float(py)) * float(self.cell_size_m)
                        if abs(float(dx)) + abs(float(dy)) > 1e-9:
                            psi_override = wrap_angle_rad(math.atan2(float(dy), float(dx)))

                if chosen_ref_idx is None:
                    costs = self._curriculum_start_costs_m
                    idxs = np.nonzero((costs >= float(lo)) & (costs <= float(hi)))[0]
                    if idxs.size == 0:
                        idxs = np.nonzero(costs <= float(hi))[0]
                    if idxs.size > 0:
                        j = int(self._rng.choice(idxs))
                        start_xy = (int(self._curriculum_start_xy[j, 0]), int(self._curriculum_start_xy[j, 1]))
                        ha_start_xy = (int(start_xy[0]), int(start_xy[1]))
                        ha_progress_idx = 0

        # 确定本回合起点（使用课程学习/随机化时可能与标准起点不同）。
        self.start_xy = (int(start_xy[0]), int(start_xy[1]))

        self._x_m = float(start_xy[0]) * self.cell_size_m
        self._y_m = float(start_xy[1]) * self.cell_size_m
        dx = float(self.goal_xy[0] - start_xy[0]) * self.cell_size_m
        dy = float(self.goal_xy[1] - start_xy[1]) * self.cell_size_m
        psi = wrap_angle_rad(math.atan2(dy, dx))
        # 回退：若 atan2 航向导致碰撞，改用代价梯度方向。
        if psi_override is None:
            _od_chk, coll_chk = self._od_and_collision_at_pose_m(
                float(self._x_m), float(self._y_m), float(psi),
            )
            if bool(coll_chk):
                psi_grad = self._heading_from_cost_gradient(
                    int(start_xy[0]), int(start_xy[1]),
                )
                if psi_grad is not None:
                    psi = psi_grad
        if psi_override is not None:
            psi = float(psi_override)
        self._psi_rad = float(psi)
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._prev_delta_dot = 0.0
        self._prev_a = 0.0
        self._last_od_m, self._last_collision = self._od_and_collision_m()
        self._stuck_pos_history = [(float(self._x_m), float(self._y_m))]
        self._ha_start_xy = (int(ha_start_xy[0]), int(ha_start_xy[1]))
        self._ha_progress_idx = int(ha_progress_idx)

        obs = self._observe()
        info = {"agent_xy": self._agent_xy_for_plot(), "pose_m": (self._x_m, self._y_m, self._psi_rad)}
        return obs, info

    def _step_with_controls(self, *, delta_dot: float, a: float):
        self._steps += 1

        delta_dot = float(delta_dot)
        a = float(a)
        prev_delta_dot = float(self._prev_delta_dot)
        prev_a = float(self._prev_a)

        x_before = float(self._x_m)
        y_before = float(self._y_m)
        # 进度塑形使用考虑安全距离的 goal distance field（有助于绕障）。
        cost_before = self._cost_to_goal_pose_m(x_before, y_before, float(self._psi_rad))
        d_goal_before = self._distance_to_goal_m()
        delta_before = float(self._delta_rad)

        x_next, y_next, psi_next, v_next, delta_next = bicycle_integrate_one_step(
            x_m=self._x_m,
            y_m=self._y_m,
            psi_rad=self._psi_rad,
            v_m_s=self._v_m_s,
            delta_rad=self._delta_rad,
            delta_dot_rad_s=delta_dot,
            a_m_s2=a,
            params=self.model,
        )

        self._x_m, self._y_m, self._psi_rad, self._v_m_s, self._delta_rad = (
            x_next,
            y_next,
            psi_next,
            v_next,
            delta_next,
        )
        self._last_od_m, self._last_collision = self._od_and_collision_m()

        cost_after = self._cost_to_goal_pose_m(self._x_m, self._y_m, float(self._psi_rad))
        d_goal_after = self._distance_to_goal_m()
        alpha = self._goal_relative_angle_rad()
        reached = (d_goal_after <= self.goal_tolerance_m) and (abs(alpha) <= self.goal_angle_tolerance_rad) and (abs(float(self._v_m_s)) <= self.goal_speed_tol_m_s)

        collision = bool(self._last_collision)
        od_m = float(self._last_od_m)
        truncated = self._steps >= self.max_steps
        terminated = bool(collision or reached)

        # 卡住检测（防止原地转向抖动 / 永久停止）。
        #
        # 使用滑动窗口位移而非单步位移：dt=0.05s 时车辆低速下每步可能合理移动 <2cm，
        # 单步阈值会导致误判卡住。
        stuck = False
        if not (terminated or truncated):
            self._stuck_pos_history.append((float(self._x_m), float(self._y_m)))
            max_hist = int(self.stuck_steps) + 1
            if len(self._stuck_pos_history) > max_hist:
                self._stuck_pos_history.pop(0)

            if len(self._stuck_pos_history) >= max_hist and abs(float(self._v_m_s)) < float(self.stuck_min_speed_m_s):
                x0, y0 = self._stuck_pos_history[0]
                x1, y1 = self._stuck_pos_history[-1]
                disp = float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
                if disp < float(self.stuck_min_disp_m):
                    stuck = True
                    terminated = True

        reward = 0.0
        # 进度奖励（接近目标）
        if math.isfinite(cost_before) and math.isfinite(cost_after):
            reward += self.reward_k_p * float(cost_before - cost_after)
        else:
            reward += self.reward_k_p * float(d_goal_before - d_goal_after)
        # 时间惩罚：每步固定惩罚，鼓励快速到达。
        reward -= self.reward_k_t
        # 效率惩罚：惩罚无效运动（行驶了距离但无测地线进度）。
        if self.reward_k_eff > 0.0:
            dist_traveled = math.hypot(float(self._x_m) - x_before, float(self._y_m) - y_before)
            if math.isfinite(cost_before) and math.isfinite(cost_after):
                progress = float(cost_before - cost_after)
            else:
                progress = float(d_goal_before - d_goal_after)
            if dist_traveled > 1e-6:
                eff = max(0.0, progress) / dist_traveled  # 1.0 = perfect, 0 = wasted
                reward -= self.reward_k_eff * max(0.0, 1.0 - eff)
        # 转向平滑性惩罚
        reward -= self.reward_k_delta * float(delta_next - delta_before) ** 2
        # 加速度平滑性惩罚：不应阻止从静止起步，因此按速度缩放。
        v_scale = (float(v_next) / float(self.model.v_max_m_s)) ** 2
        reward -= self.reward_k_a * float(a - prev_a) ** 2 * float(v_scale)
        # 曲率 / 大转向角惩罚
        reward -= self.reward_k_kappa * float(math.tan(delta_next) ** 2)
        # 基于安全距离的奖励塑形。已碰撞时跳过，避免叠加过大惩罚。
        if not collision:
            od_pos = max(0.0, float(od_m))

            # 近障碍物惩罚（基于 OD 安全距离）。
            if od_pos < self.safe_distance_m:
                obs_term = (1.0 / (od_pos + self.reward_eps)) - (1.0 / (self.safe_distance_m + self.reward_eps))
                obs_pen = float(self.reward_k_o) * float(obs_term)
                reward -= min(float(self.reward_obs_max), float(obs_pen))

            # 森林环境近障碍物速度耦合 + 可选软速度上限。
            if od_pos < self.safe_speed_distance_m:
                # 速度耦合项（安全距离小时惩罚高速）。
                reward -= self.reward_k_v * ((self.safe_speed_distance_m - od_pos) / self.safe_speed_distance_m) * (
                    float(v_next) / float(self.model.v_max_m_s)
                ) ** 2

                # 软速度上限（可选，在狭窄走廊中稳定行驶）。
                v_cap = float(self.model.v_max_m_s) * float(
                    np.clip(float(od_pos) / float(self.safe_speed_distance_m), 0.0, 1.0)
                )
                dv = max(0.0, float(v_next) - float(v_cap))
                reward -= self.reward_k_c * float(dv) ** 2

        # 目标接近塑形（进入目标区域附近时的每步奖励）。
        if self.reward_k_goal > 0.0:
            _shaping_r = 1.5 * self.goal_tolerance_m
            if d_goal_after < _shaping_r:
                reward += self.reward_k_goal * (1.0 - d_goal_after / _shaping_r)
                # 目标附近速度惩罚：惩罚高速以鼓励减速停车。
                if self.goal_speed_tol_m_s < 900.0:
                    _v_ratio = abs(float(self._v_m_s)) / float(self.model.v_max_m_s)
                    reward -= self.reward_k_goal * _v_ratio

        # 终止奖励/惩罚
        if collision:
            reward -= 200.0
        elif reached:
            if self.reward_k_goal > 0.0:
                _prox = max(0.0, 1.0 - d_goal_after / self.goal_tolerance_m)
                reward += 350.0 + 50.0 * _prox
            else:
                reward += 400.0
        if stuck:
            reward -= float(self.stuck_penalty)

        # 马尔可夫性：下一状态携带的"上一动作"= 本步执行的动作。
        self._prev_delta_dot = float(delta_dot)
        self._prev_a = float(a)

        obs = self._observe()
        info = {
            "agent_xy": self._agent_xy_for_plot(),
            "pose_m": (self._x_m, self._y_m, self._psi_rad),
            "collision": bool(collision),
            "reached": bool(reached),
            "stuck": bool(stuck),
            "od_m": float(od_m),
            "d_goal_m": float(d_goal_after),
            "alpha_rad": float(alpha),
            "v_m_s": float(self._v_m_s),
            "delta_rad": float(self._delta_rad),
            "steps": int(self._steps),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step(self, action: int):
        a_id = int(action)
        delta_dot = float(self.action_table[a_id, 0])
        a = float(self.action_table[a_id, 1])
        return self._step_with_controls(delta_dot=delta_dot, a=a)

    def step_continuous(self, *, delta_dot_rad_s: float, a_m_s2: float):
        """step() 的连续控制变体（使用相同的动力学/碰撞/终止逻辑）。

        用于在森林环境中评估连续控制器（如 MPC），无需强制通过 DQN 使用的
        离散 action_table 接口。
        """
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)
        delta_dot = float(np.clip(float(delta_dot_rad_s), -dd_max, +dd_max))
        a = float(np.clip(float(a_m_s2), -a_max, +a_max))
        return self._step_with_controls(delta_dot=delta_dot, a=a)

    def step_continuous_direct(self, *, delta_rad: float, v_m_s: float):
        """传统 MPC 控制接口：直接指定目标转向角 δ 和目标速度 v。

        内部将 (δ, v) 转换为 (δ̇, a) 后调用 _step_with_controls。
        转换公式：δ̇ = (δ_target - δ_current) / dt，a = (v_target - v_current) / dt，
        并裁剪到执行器限幅。
        """
        dt = float(self.model.dt)
        delta_max = float(self.model.delta_max_rad)
        v_max = float(self.model.v_max_m_s)
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)

        # 裁剪目标值到物理限幅
        delta_target = float(np.clip(float(delta_rad), -delta_max, +delta_max))
        v_target = float(np.clip(float(v_m_s), 0.0, +v_max))

        # 转换为速率控制量
        delta_dot = float(np.clip((delta_target - float(self._delta_rad)) / dt, -dd_max, +dd_max))
        a = float(np.clip((v_target - float(self._v_m_s)) / dt, -a_max, +a_max))

        return self._step_with_controls(delta_dot=delta_dot, a=a)

    def _agent_xy_for_plot(self) -> tuple[float, float]:
        return (float(self._x_m) / self.cell_size_m, float(self._y_m) / self.cell_size_m)

    def _circle_centers_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        c = math.cos(float(self._psi_rad))
        s = math.sin(float(self._psi_rad))
        c1 = (float(self._x_m) + c * float(self.footprint.x1_m), float(self._y_m) + s * float(self.footprint.x1_m))
        c2 = (float(self._x_m) + c * float(self.footprint.x2_m), float(self._y_m) + s * float(self.footprint.x2_m))
        return c1, c2

    def _dist_at_m(self, x_m: float, y_m: float) -> float:
        xi = float(x_m) / self.cell_size_m
        yi = float(y_m) / self.cell_size_m
        return bilinear_sample_2d(self._dist_m, x=xi, y=yi, default=0.0)

    @staticmethod
    def _wrap_angle_rad_np(x: np.ndarray) -> np.ndarray:
        return (np.remainder(np.asarray(x, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi).astype(
            np.float64, copy=False
        )

    def _dist_at_m_vec(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        xi = np.asarray(x_m, dtype=np.float64) / float(self.cell_size_m)
        yi = np.asarray(y_m, dtype=np.float64) / float(self.cell_size_m)
        return bilinear_sample_2d_vec(self._dist_m, x=xi, y=yi, default=0.0)

    def _od_and_collision_m(self) -> tuple[float, bool]:
        if not self._in_world_bounds(self._x_m, self._y_m):
            return -float("inf"), True
        c1, c2 = self._circle_centers_m()
        d1 = self._dist_at_m(c1[0], c1[1])
        d2 = self._dist_at_m(c2[0], c2[1])
        r = float(self.footprint.radius_m)
        od_m = min(d1 - r, d2 - r)
        # EDT 测量的是中心到中心的距离；加上半栅格边距以补偿障碍物栅格
        # 从中心延伸 0.5*cell_size 的范围。
        r_col = r + self._half_cell_m
        collision = (d1 <= r_col) or (d2 <= r_col)
        return float(od_m), bool(collision)

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        if not self._in_world_bounds(x_m, y_m):
            return -float("inf"), True

        c = math.cos(float(psi_rad))
        s = math.sin(float(psi_rad))
        c1x = float(x_m) + c * float(self.footprint.x1_m)
        c1y = float(y_m) + s * float(self.footprint.x1_m)
        c2x = float(x_m) + c * float(self.footprint.x2_m)
        c2y = float(y_m) + s * float(self.footprint.x2_m)

        d1 = self._dist_at_m(c1x, c1y)
        d2 = self._dist_at_m(c2x, c2y)
        r = float(self.footprint.radius_m)
        od_m = min(float(d1) - r, float(d2) - r)
        r_col = r + self._half_cell_m
        collision = (float(d1) <= r_col) or (float(d2) <= r_col)
        return float(od_m), bool(collision)

    def _od_and_collision_at_pose_m_vec(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        xv = np.asarray(x_m, dtype=np.float64)
        yv = np.asarray(y_m, dtype=np.float64)
        psiv = np.asarray(psi_rad, dtype=np.float64)

        max_x = float(self._width - 1) * float(self.cell_size_m)
        max_y = float(self._height - 1) * float(self.cell_size_m)
        in_bounds = (xv >= 0.0) & (xv <= max_x) & (yv >= 0.0) & (yv <= max_y)

        c = np.cos(psiv)
        s = np.sin(psiv)
        c1x = xv + c * float(self.footprint.x1_m)
        c1y = yv + s * float(self.footprint.x1_m)
        c2x = xv + c * float(self.footprint.x2_m)
        c2y = yv + s * float(self.footprint.x2_m)

        d1 = self._dist_at_m_vec(c1x, c1y)
        d2 = self._dist_at_m_vec(c2x, c2y)
        r = float(self.footprint.radius_m)
        od = np.minimum(d1 - r, d2 - r)
        r_col = r + self._half_cell_m
        coll = (d1 <= r_col) | (d2 <= r_col)

        od = np.where(in_bounds, od, -float("inf")).astype(np.float64, copy=False)
        coll = np.where(in_bounds, coll, True).astype(np.bool_, copy=False)
        return od, coll

    def _cost_to_goal_at_m_vec(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        xi = np.asarray(x_m, dtype=np.float64) / float(self.cell_size_m)
        yi = np.asarray(y_m, dtype=np.float64) / float(self.cell_size_m)
        return bilinear_sample_2d_finite_vec(
            self._cost_to_goal_m,
            x=xi,
            y=yi,
            fill_value=float(self._cost_fill_m),
        )

    def _cost_to_goal_pose_m_vec(self, x_m: np.ndarray, y_m: np.ndarray, psi_rad: np.ndarray) -> np.ndarray:
        xv = np.asarray(x_m, dtype=np.float64)
        yv = np.asarray(y_m, dtype=np.float64)
        psiv = np.asarray(psi_rad, dtype=np.float64)
        c = np.cos(psiv)
        s = np.sin(psiv)
        c1x = xv + c * float(self.footprint.x1_m)
        c1y = yv + s * float(self.footprint.x1_m)
        c2x = xv + c * float(self.footprint.x2_m)
        c2y = yv + s * float(self.footprint.x2_m)
        c1 = self._cost_to_goal_at_m_vec(c1x, c1y)
        c2 = self._cost_to_goal_at_m_vec(c2x, c2y)
        return np.maximum(c1, c2).astype(np.float64, copy=False)

    def _bicycle_integrate_one_step_vec(
        self,
        *,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
        v_m_s: np.ndarray,
        delta_rad: np.ndarray,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dt = float(self.model.dt)
        wheelbase = float(self.model.wheelbase_m)
        v_max = float(self.model.v_max_m_s)
        delta_max = float(self.model.delta_max_rad)

        v_next = np.clip(
            np.asarray(v_m_s, dtype=np.float64) + np.asarray(a_m_s2, dtype=np.float64) * dt,
            -float(v_max),
            float(v_max),
        )

        delta_unclipped = np.asarray(delta_rad, dtype=np.float64) + np.asarray(delta_dot_rad_s, dtype=np.float64) * dt
        delta_next = np.clip(delta_unclipped, -float(delta_max), +float(delta_max))

        psi = np.asarray(psi_rad, dtype=np.float64)
        x_next = np.asarray(x_m, dtype=np.float64) + v_next * np.cos(psi) * dt
        y_next = np.asarray(y_m, dtype=np.float64) + v_next * np.sin(psi) * dt
        psi_next = self._wrap_angle_rad_np(psi + (v_next / wheelbase) * np.tan(delta_next) * dt)
        return x_next, y_next, psi_next, v_next, delta_next

    def _rollout_constant_actions_end_state(
        self,
        *,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = max(1, int(horizon_steps))
        delta_dot = np.asarray(delta_dot_rad_s, dtype=np.float64).reshape(-1)
        accel = np.asarray(a_m_s2, dtype=np.float64).reshape(-1)
        if delta_dot.shape != accel.shape:
            raise ValueError("delta_dot_rad_s and a_m_s2 must have the same shape")

        n = int(delta_dot.size)
        x = np.full((n,), float(self._x_m), dtype=np.float64)
        y = np.full((n,), float(self._y_m), dtype=np.float64)
        psi = np.full((n,), float(self._psi_rad), dtype=np.float64)
        v = np.full((n,), float(self._v_m_s), dtype=np.float64)
        delta = np.full((n,), float(self._delta_rad), dtype=np.float64)

        min_od = np.full((n,), float("inf"), dtype=np.float64)
        coll = np.zeros((n,), dtype=np.bool_)
        reached = np.zeros((n,), dtype=np.bool_)
        active = np.ones((n,), dtype=np.bool_)

        gx_m = float(self.goal_xy[0]) * float(self.cell_size_m)
        gy_m = float(self.goal_xy[1]) * float(self.cell_size_m)
        tol_m = float(self.goal_tolerance_m)

        for _ in range(h):
            if not bool(active.any()):
                break

            x1, y1, psi1, v1, delta1 = self._bicycle_integrate_one_step_vec(
                x_m=x,
                y_m=y,
                psi_rad=psi,
                v_m_s=v,
                delta_rad=delta,
                delta_dot_rad_s=delta_dot,
                a_m_s2=accel,
            )
            # 冻结已终止的 rollout（到达/碰撞），使后续步不影响掩码。
            x = np.where(active, x1, x)
            y = np.where(active, y1, y)
            psi = np.where(active, psi1, psi)
            v = np.where(active, v1, v)
            delta = np.where(active, delta1, delta)

            od, coll_step = self._od_and_collision_at_pose_m_vec(x, y, psi)
            min_od = np.where(active, np.minimum(min_od, od), min_od)
            coll_now = coll_step & active
            coll |= coll_now

            d_goal_m = np.hypot(float(gx_m) - x, float(gy_m) - y)
            reached_now = (d_goal_m <= float(tol_m)) & active & (~coll_now)
            reached |= reached_now

            active &= ~(coll_now | reached_now)

        return x, y, psi, v, min_od, coll, reached

    def _rollout_direct_controls_sequence(
        self,
        *,
        delta_rad_seq: np.ndarray,
        v_m_s_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """传统 MPC 用：给定 (δ, v) 控制序列，前向仿真并返回轨迹。

        参数：
            delta_rad_seq: shape (H,) — 每步目标转向角
            v_m_s_seq:     shape (H,) — 每步目标速度

        返回：
            x_traj, y_traj, psi_traj, v_traj: shape (H+1,) 含初始状态
            min_od: 全程最小障碍物距离
            collision: 是否发生碰撞
        """
        dt = float(self.model.dt)
        wheelbase = float(self.model.wheelbase_m)
        delta_max = float(self.model.delta_max_rad)
        v_max = float(self.model.v_max_m_s)
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)

        delta_seq = np.clip(np.asarray(delta_rad_seq, dtype=np.float64), -delta_max, +delta_max)
        v_seq = np.clip(np.asarray(v_m_s_seq, dtype=np.float64), 0.0, v_max)
        h = int(delta_seq.shape[0])

        # 轨迹数组（含初始状态）
        x_traj = np.empty(h + 1, dtype=np.float64)
        y_traj = np.empty(h + 1, dtype=np.float64)
        psi_traj = np.empty(h + 1, dtype=np.float64)
        v_traj = np.empty(h + 1, dtype=np.float64)

        x_traj[0] = float(self._x_m)
        y_traj[0] = float(self._y_m)
        psi_traj[0] = float(self._psi_rad)
        v_traj[0] = float(self._v_m_s)
        cur_delta = float(self._delta_rad)

        min_od = float("inf")
        collision = False

        for k in range(h):
            # (δ, v) → (δ̇, a)，裁剪到执行器限幅
            delta_dot = np.clip((float(delta_seq[k]) - cur_delta) / dt, -dd_max, +dd_max)
            accel = np.clip((float(v_seq[k]) - v_traj[k]) / dt, -a_max, +a_max)

            # 自行车模型积分
            v_next = np.clip(v_traj[k] + accel * dt, -v_max, v_max)
            delta_next = np.clip(cur_delta + delta_dot * dt, -delta_max, +delta_max)
            psi_k = psi_traj[k]
            x_next = x_traj[k] + v_next * math.cos(psi_k) * dt
            y_next = y_traj[k] + v_next * math.sin(psi_k) * dt
            psi_next = psi_k + (v_next / wheelbase) * math.tan(delta_next) * dt

            x_traj[k + 1] = x_next
            y_traj[k + 1] = y_next
            psi_traj[k + 1] = psi_next
            v_traj[k + 1] = v_next
            cur_delta = delta_next

            # 碰撞检测
            od, coll_step = self._od_and_collision_at_pose_m_vec(
                np.array([x_next]), np.array([y_next]), np.array([psi_next])
            )
            min_od = min(min_od, float(od[0]))
            if bool(coll_step[0]):
                collision = True
                break

        return x_traj, y_traj, psi_traj, v_traj, min_od, collision

    def _fallback_action_short_rollout(
        self,
        *,
        horizon_steps: int,
        min_od_m: float = 0.0,
    ) -> int:
        """基于目标距离图的贪心动作选择器。

        对所有离散动作做短视野恒定动作 rollout，选取末端 goal distance
        最低且全程无碰撞的动作。同时作为 cost_to_go 专家的主逻辑
        和 Hybrid A* 专家不可用时的回退策略。
        """

        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, _v, min_od, coll, reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        cost1 = self._cost_to_goal_pose_m_vec(x, y, psi)

        ok = (~coll) & (min_od >= float(min_od_thr)) & np.isfinite(cost1)
        if bool(ok.any()):
            ok_reached = ok & reached
            idx = np.nonzero(ok_reached if bool(ok_reached.any()) else ok)[0]
            costs = cost1[idx]
            ods = min_od[idx]
            best_cost = float(np.min(costs))
            cand = idx[costs <= float(best_cost) + 1e-9]
            if cand.size == 0:
                cand = idx[int(np.argmin(costs))]
                return int(cand)
            best = int(cand[int(np.argmax(min_od[cand]))])
            return int(best)

        # 最后手段：选择单步安全距离最大的动作（即使仍可能碰撞）。
        x0 = float(self._x_m)
        y0 = float(self._y_m)
        psi0 = float(self._psi_rad)
        v0 = float(self._v_m_s)
        delta0 = float(self._delta_rad)

        best_action = 0
        best_od = -float("inf")
        for a_id in range(int(self.action_table.shape[0])):
            delta_dot = float(self.action_table[a_id, 0])
            a = float(self.action_table[a_id, 1])
            x, y, psi, _v, _delta = bicycle_integrate_one_step(
                x_m=x0,
                y_m=y0,
                psi_rad=psi0,
                v_m_s=v0,
                delta_rad=delta0,
                delta_dot_rad_s=delta_dot,
                a_m_s2=a,
                params=self.model,
            )
            od, _coll = self._od_and_collision_at_pose_m(x, y, psi)
            if float(od) > float(best_od):
                best_od = float(od)
                best_action = int(a_id)
        return int(best_action)

    def _hybrid_astar_path(self, *, start_xy: tuple[int, int], timeout_s: float = 5.0, max_nodes: int = 200_000) -> list[tuple[float, float]]:
        key = (int(start_xy[0]), int(start_xy[1]), int(self.goal_xy[0]), int(self.goal_xy[1]))
        cached = self._ha_path_cache.get(key)
        if cached is not None:
            return cached

        # 快速路径：加载预计算的 Hybrid A* 参考路径（针对固定森林起点）。
        # 在固定种子下结果确定，避免训练/推理时的规划开销。
        if (
            self.map_spec.name.startswith("forest_")
            and (int(start_xy[0]), int(start_xy[1])) == (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
            and (int(self.goal_xy[0]), int(self.goal_xy[1])) == (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))
        ):
            pre = Path(__file__).resolve().parent / "maps" / "precomputed" / f"{self.map_spec.name}_hybrid_astar_path.json"
            if pre.exists():
                try:
                    payload = json.loads(pre.read_text(encoding="utf-8"))
                    pts = payload.get("path_xy_cells")
                    if bool(payload.get("success")) and isinstance(pts, list) and len(pts) >= 2:
                        path = [(float(p[0]), float(p[1])) for p in pts]
                        h, w = self._grid.shape
                        sx, sy = int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1])
                        gx, gy = int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1])
                        tol_cells = float(self.goal_tolerance_m) / float(self.cell_size_m)

                        def cell_free(xc: float, yc: float) -> bool:
                            xi = int(round(float(xc)))
                            yi = int(round(float(yc)))
                            if not (0 <= xi < w and 0 <= yi < h):
                                return False
                            return int(self._grid[yi, xi]) == 0

                        start_ok = (float(path[0][0]) - float(sx)) ** 2 + (float(path[0][1]) - float(sy)) ** 2 <= 4.0
                        goal_ok = (float(path[-1][0]) - float(gx)) ** 2 + (float(path[-1][1]) - float(gy)) ** 2 <= (tol_cells + 2.0) ** 2
                        path_ok = bool(start_ok) and bool(goal_ok) and all(cell_free(x, y) for x, y in path)
                        if path_ok:
                            self._ha_path_cache[key] = path
                            return path
                except Exception:
                    pass

        try:
            from ugv_dqn.baselines.pathplan import (
                default_ackermann_params,
                forest_two_circle_footprint,
                grid_map_from_obstacles,
                plan_hybrid_astar,
            )
        except Exception:
            self._ha_path_cache[key] = []
            return []

        grid_map = grid_map_from_obstacles(grid_y0_bottom=self._grid, cell_size_m=float(self.cell_size_m))
        params = default_ackermann_params(
            wheelbase_m=float(self.model.wheelbase_m),
            delta_max_rad=float(self.model.delta_max_rad),
            v_max_m_s=float(self.model.v_max_m_s),
        )
        footprint = forest_two_circle_footprint()

        res = plan_hybrid_astar(
            grid_map=grid_map,
            footprint=footprint,
            params=params,
            start_xy=(int(start_xy[0]), int(start_xy[1])),
            goal_xy=(int(self.goal_xy[0]), int(self.goal_xy[1])),
            goal_theta_rad=0.0,
            start_theta_rad=None,
            goal_xy_tol_m=float(self.goal_tolerance_m),
            goal_theta_tol_rad=float(math.pi),
            timeout_s=float(timeout_s),
            max_nodes=int(max_nodes),
        )
        path = list(res.path_xy_cells) if res.success else []
        self._ha_path_cache[key] = path
        return path

    def expert_action_hybrid_astar(
        self,
        *,
        lookahead_points: int = 3,
        horizon_steps: int = 4,
        w_target: float = 0.5,
        w_heading: float = 0.4,
        w_clearance: float = 0.2,
        w_speed: float = 0.0,
    ) -> int:
        """Hybrid A* 引导专家（用于 DQfD 演示 / 引导探索）。

        每个回合开始时计算一次 Hybrid A* 参考路径（按起始栅格缓存），然后通过
        纯追踪（pure-pursuit）风格的转向目标 + 短视野安全掩码下的离散控制选择来跟踪。
        """

        path = self._hybrid_astar_path(start_xy=self._ha_start_xy)
        if len(path) < 2:
            return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=0.0)

        x_cells = float(self._x_m) / float(self.cell_size_m)
        y_cells = float(self._y_m) / float(self.cell_size_m)

        # 在上一索引附近的有限窗口内寻找最近路径点索引。
        start_i = max(0, int(self._ha_progress_idx) - 25)
        end_i = min(len(path), int(self._ha_progress_idx) + 250)
        if end_i <= start_i:
            start_i, end_i = 0, len(path)
        best_i = start_i
        best_d2 = float("inf")
        for i in range(start_i, end_i):
            px, py = path[i]
            d2 = (float(px) - x_cells) ** 2 + (float(py) - y_cells) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        self._ha_progress_idx = int(best_i)

        la = max(1, int(lookahead_points))
        tgt_i = min(int(best_i) + la, len(path) - 1)
        tx_cells, ty_cells = path[tgt_i]
        tx_m = float(tx_cells) * float(self.cell_size_m)
        ty_m = float(ty_cells) * float(self.cell_size_m)

        h = max(1, int(horizon_steps))
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, v, min_od, coll, _reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )

        cost = self._cost_to_goal_pose_m_vec(x, y, psi)
        dist_tgt = np.hypot(float(tx_m) - x, float(ty_m) - y)
        tgt_heading = np.arctan2(float(ty_m) - y, float(tx_m) - x)
        heading_err = self._wrap_angle_rad_np(tgt_heading - psi)

        score = -cost
        score += -float(w_target) * dist_tgt - float(w_heading) * np.abs(heading_err)
        score += float(w_clearance) * min_od

        if float(w_speed) != 0.0:
            v_max = float(self.model.v_max_m_s)
            score += float(w_speed) * (v / max(1e-9, float(v_max)))

        invalid = coll | (~np.isfinite(score)) | (~np.isfinite(cost))
        score = np.where(invalid, -float("inf"), score)
        best_action = int(np.argmax(score))
        if not math.isfinite(float(score[best_action])):
            return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=0.0)

        return int(best_action)

    def expert_action_cost_to_go(
        self,
        *,
        horizon_steps: int = 15,
        min_od_m: float = 0.0,
    ) -> int:
        """基于 goal distance field 短视野 rollout 的轻量级专家。"""
        return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=float(min_od_m))

    def _rollout_constant_action_metrics(
        self,
        a_id: int,
        *,
        horizon_steps: int,
    ) -> tuple[float, float, float, bool, bool]:
        """模拟短视野内恒定离散动作。

        返回: (末端 goal distance, 末端速度, 视野内最小 OD, 是否碰撞, 是否到达目标)。
        """

        h = max(1, int(horizon_steps))
        a_id = int(a_id)

        delta_dot = float(self.action_table[a_id, 0])
        a = float(self.action_table[a_id, 1])

        x = float(self._x_m)
        y = float(self._y_m)
        psi = float(self._psi_rad)
        v = float(self._v_m_s)
        delta = float(self._delta_rad)

        gx_m = float(self.goal_xy[0]) * float(self.cell_size_m)
        gy_m = float(self.goal_xy[1]) * float(self.cell_size_m)
        tol_m = float(self.goal_tolerance_m)

        min_od = float("inf")
        for _ in range(h):
            x, y, psi, v, delta = bicycle_integrate_one_step(
                x_m=x,
                y_m=y,
                psi_rad=psi,
                v_m_s=v,
                delta_rad=delta,
                delta_dot_rad_s=delta_dot,
                a_m_s2=a,
                params=self.model,
            )
            od, coll = self._od_and_collision_at_pose_m(x, y, psi)
            min_od = min(float(min_od), float(od))
            if coll:
                return float("inf"), float(v), float(min_od), True, False

            if float(math.hypot(float(gx_m) - float(x), float(gy_m) - float(y))) <= float(tol_m):
                cost = float(self._cost_to_goal_pose_m(x, y, psi))
                return float(cost), float(v), float(min_od), False, True

        cost = float(self._cost_to_goal_pose_m(x, y, psi))
        return float(cost), float(v), float(min_od), False, False

    def is_action_safe(
        self,
        a_id: int,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
    ) -> bool:
        _cost, _v, min_od, coll, _reached = self._rollout_constant_action_metrics(
            int(a_id), horizon_steps=int(horizon_steps)
        )
        if bool(coll):
            return False
        return float(min_od) >= float(min_od_m)

    def is_action_admissible(
        self,
        a_id: int,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
        min_progress_m: float = 1e-4,
        allow_reverse: bool = True,
    ) -> bool:
        cost0 = float(self._cost_to_goal_pose_m(float(self._x_m), float(self._y_m), float(self._psi_rad)))
        if not math.isfinite(cost0):
            return True

        # 进度在短视野恒定动作 rollout 末端判断，安全性（碰撞/安全距离）在整个视野内判断。
        h = max(1, int(horizon_steps))
        cost1, v_end, min_od, coll, reached = self._rollout_constant_action_metrics(int(a_id), horizon_steps=h)
        if bool(coll):
            return False
        if float(min_od) < float(min_od_m):
            return False
        if bool(reached):
            return True
        if not math.isfinite(cost1):
            return False
        if float(cost0 - cost1) >= float(min_progress_m):
            return True

        # 仅在相同短视野约束下无前进动作时才允许倒车/后退。
        # 避免目标附近的退化行为（策略持续选择后退/停止动作导致卡住终止）。
        if bool(allow_reverse):
            reverse_v_min = 0.10
            if float(v_end) < -float(reverse_v_min):
                prog_mask = self.admissible_action_mask(
                    horizon_steps=h,
                    min_od_m=float(min_od_m),
                    min_progress_m=float(min_progress_m),
                    fallback_to_safe=False,
                    allow_reverse=False,
                )
                if not bool(prog_mask.any()):
                    return True
        return False

    def safe_action_mask(
        self,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
    ) -> np.ndarray:
        """返回短视野内保持无碰撞的动作布尔掩码。"""
        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        _x, _y, _psi, _v, min_od, coll, _reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        out = (~coll) & (min_od >= float(min_od_thr))
        return out.astype(np.bool_, copy=False)

    def admissible_action_mask(
        self,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
        min_progress_m: float = 1e-4,
        fallback_to_safe: bool = True,
        allow_reverse: bool = True,
    ) -> np.ndarray:
        """掩码安全且在 goal distance 上有进度的动作（可选允许倒车）。"""

        cost0 = float(self._cost_to_goal_pose_m(float(self._x_m), float(self._y_m), float(self._psi_rad)))
        out = np.zeros((int(self.action_table.shape[0]),), dtype=np.bool_)
        if not math.isfinite(cost0):
            out[:] = True
            return out

        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        min_prog = float(min_progress_m)

        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, v_end, min_od, coll, reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        cost1 = self._cost_to_goal_pose_m_vec(x, y, psi)

        safe = (~coll) & (min_od >= float(min_od_thr)) & np.isfinite(cost1)
        prog = ((float(cost0) - cost1) >= float(min_prog)) | reached
        out = safe & prog
        if bool(allow_reverse) and not bool(out.any()):
            # 仅在无前进动作时暴露倒车动作。
            reverse_v_min = 0.10
            out = safe & (v_end < -float(reverse_v_min))

        # 回退：若所有动作都被过滤，保留无碰撞的安全动作。
        if bool(fallback_to_safe) and not bool(out.any()):
            out = (~coll) & (min_od >= float(min_od_thr))
        return out.astype(np.bool_, copy=False)

    def _observe(self) -> np.ndarray:
        # 归一化 (x,y) + 目标 (x,y) 到 [-1,1]。
        max_x = max(1e-6, float(self._width - 1) * self.cell_size_m)
        max_y = max(1e-6, float(self._height - 1) * self.cell_size_m)
        ax_n = 2.0 * (float(self._x_m) / float(max_x)) - 1.0
        ay_n = 2.0 * (float(self._y_m) / float(max_y)) - 1.0
        gx_n = 2.0 * ((float(self.goal_xy[0]) * self.cell_size_m) / float(max_x)) - 1.0
        gy_n = 2.0 * ((float(self.goal_xy[1]) * self.cell_size_m) / float(max_y)) - 1.0

        # 标量特征
        sin_psi = float(math.sin(float(self._psi_rad)))
        cos_psi = float(math.cos(float(self._psi_rad)))
        v_n = float(self._v_m_s) / float(self.model.v_max_m_s)
        delta_lim = float(self.model.delta_max_rad)
        delta_n = 0.0 if abs(delta_lim) < 1e-9 else float(self._delta_rad) / float(delta_lim)
        cost = self._cost_to_goal_pose_m(self._x_m, self._y_m, float(self._psi_rad))
        if math.isfinite(cost):
            cost01 = float(cost) / max(1e-6, float(self._cost_norm_m))
        else:
            cost01 = float(self._distance_to_goal_m()) / max(1e-6, float(self._diag_m))
        cost_n = 2.0 * float(np.clip(cost01, 0.0, 1.0)) - 1.0

        alpha_n = float(self._goal_relative_angle_rad()) / math.pi
        od01 = min(self.od_cap_m, max(0.0, float(self._last_od_m))) / float(self.od_cap_m)
        od_n = 2.0 * float(np.clip(od01, 0.0, 1.0)) - 1.0

        # 钳位到稳定范围。
        ax_n = float(np.clip(ax_n, -1.0, 1.0))
        ay_n = float(np.clip(ay_n, -1.0, 1.0))
        gx_n = float(np.clip(gx_n, -1.0, 1.0))
        gy_n = float(np.clip(gy_n, -1.0, 1.0))
        sin_psi = float(np.clip(sin_psi, -1.0, 1.0))
        cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
        v_n = float(np.clip(v_n, -1.0, 1.0))
        delta_n = float(np.clip(delta_n, -1.0, 1.0))
        cost_n = float(np.clip(cost_n, -1.0, 1.0))
        alpha_n = float(np.clip(alpha_n, -1.0, 1.0))
        obs = np.concatenate(
            [
                np.array(
                    [ax_n, ay_n, gx_n, gy_n, sin_psi, cos_psi, v_n, delta_n, cost_n, alpha_n, od_n],
                    dtype=np.float32,
                ),
                self._obs_occ_flat,
                self._obs_cost_flat,
                self._obs_edt_flat,
            ]
        )
        return obs.astype(np.float32, copy=False)

    def _distance_to_goal_m(self) -> float:
        gx = float(self.goal_xy[0]) * self.cell_size_m
        gy = float(self.goal_xy[1]) * self.cell_size_m
        return float(math.hypot(gx - float(self._x_m), gy - float(self._y_m)))

    def _goal_relative_angle_rad(self) -> float:
        gx = float(self.goal_xy[0]) * self.cell_size_m
        gy = float(self.goal_xy[1]) * self.cell_size_m
        goal_heading = math.atan2(gy - float(self._y_m), gx - float(self._x_m))
        return wrap_angle_rad(float(goal_heading) - float(self._psi_rad))

    def _cost_to_goal_at_m(self, x_m: float, y_m: float) -> float:
        xi = float(x_m) / self.cell_size_m
        yi = float(y_m) / self.cell_size_m
        return bilinear_sample_2d_finite(self._cost_to_goal_m, x=xi, y=yi, fill_value=float(self._cost_fill_m))

    def _cost_to_goal_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> float:
        """整车足迹（双圆）的 goal distance。

        取两个圆心代价的最大值，使进度塑形不会鼓励
        某个圆被困/不安全的运动。
        """
        c = math.cos(float(psi_rad))
        s = math.sin(float(psi_rad))
        c1x = float(x_m) + c * float(self.footprint.x1_m)
        c1y = float(y_m) + s * float(self.footprint.x1_m)
        c2x = float(x_m) + c * float(self.footprint.x2_m)
        c2y = float(y_m) + s * float(self.footprint.x2_m)
        c1 = self._cost_to_goal_at_m(c1x, c1y)
        c2 = self._cost_to_goal_at_m(c2x, c2y)
        return float(max(float(c1), float(c2)))

    def _in_world_bounds(self, x_m: float, y_m: float) -> bool:
        max_x = float(self._width - 1) * self.cell_size_m
        max_y = float(self._height - 1) * self.cell_size_m
        return (0.0 <= float(x_m) <= max_x) and (0.0 <= float(y_m) <= max_y)

    def _sector_ray_distances_n(self) -> np.ndarray:
        """类激光雷达射线距离，归一化到 [0,1]。角度在车体坐标系下。"""
        x0 = float(self._x_m) / self.cell_size_m
        y0 = float(self._y_m) / self.cell_size_m
        max_range_cells = float(self.sensor_range_m) / self.cell_size_m
        step_cells = 0.5
        max_steps = max(1, int(math.ceil(max_range_cells / step_cells)))

        out = np.ones((self.n_sectors,), dtype=np.float32)
        for i in range(self.n_sectors):
            ang = float(self._psi_rad) + (2.0 * math.pi) * (float(i) / float(self.n_sectors))
            c = math.cos(ang)
            s = math.sin(ang)
            hit_cells = max_range_cells
            for j in range(1, max_steps + 1):
                d_cells = float(j) * step_cells
                xi = x0 + c * d_cells
                yi = y0 + s * d_cells
                ix = int(math.floor(xi + 0.5))
                iy = int(math.floor(yi + 0.5))
                if not (0 <= ix < self._width and 0 <= iy < self._height):
                    hit_cells = d_cells
                    break
                if self._grid[iy, ix] == 1:
                    hit_cells = d_cells
                    break
            out[i] = float(np.clip((hit_cells * self.cell_size_m) / float(self.sensor_range_m), 0.0, 1.0))
        return out
