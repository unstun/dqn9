from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ForestParams:
    width_cells: int = 128
    height_cells: int = 128
    cell_size_m: float = 0.1

    # 障碍物稍微远离地图边界，避免边界 EDT 截断产生大片空白边框。
    boundary_margin_m: float = 0.2

    # 树干（圆形障碍物）。
    trunk_count: int = 90
    trunk_radius_m_min: float = 0.15
    trunk_radius_m_max: float = 0.35

    # 树干表面之间的目标间隙（米）。实际间隙因树干而异。
    trunk_gap_m: float = 1.0
    trunk_gap_jitter: float = 0.25
    trunk_place_tries: int = 20_000

    # 可选的小灌木/碎石，以圆形表示（保持圆形，不做方形膨胀）。
    bush_cluster_count: int = 0
    bush_per_cluster_min: int = 3
    bush_per_cluster_max: int = 8
    bush_radius_m_min: float = 0.08
    bush_radius_m_max: float = 0.18
    bush_spread_m: float = 0.5

    start_margin_m: float = 1.0
    goal_margin_m: float = 1.0
    # 森林内部固定（确定性）的起点/终点放置位置。
    start_frac: float = 0.2
    goal_frac: float = 0.8

    max_tries: int = 200


def _bilinear_sample_2d(arr: np.ndarray, *, x: float, y: float, default: float) -> float:
    h, w = arr.shape
    if not (0.0 <= float(x) <= float(w - 1) and 0.0 <= float(y) <= float(h - 1)):
        return float(default)
    x0 = int(math.floor(float(x)))
    y0 = int(math.floor(float(y)))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x) - float(x0)
    fy = float(y) - float(y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def _two_circle_collision(
    dist_m: np.ndarray,
    *,
    x_m: float,
    y_m: float,
    psi_rad: float,
    cell_size_m: float,
    radius_m: float,
    x1_m: float,
    x2_m: float,
    eps_cell_m: float,
) -> bool:
    h, w = dist_m.shape
    max_x = float(w - 1) * float(cell_size_m)
    max_y = float(h - 1) * float(cell_size_m)
    if not (0.0 <= float(x_m) <= max_x and 0.0 <= float(y_m) <= max_y):
        return True

    c = math.cos(float(psi_rad))
    s = math.sin(float(psi_rad))
    c1x = float(x_m) + c * float(x1_m)
    c1y = float(y_m) + s * float(x1_m)
    c2x = float(x_m) + c * float(x2_m)
    c2y = float(y_m) + s * float(x2_m)

    d1 = _bilinear_sample_2d(
        dist_m, x=float(c1x) / float(cell_size_m), y=float(c1y) / float(cell_size_m), default=0.0
    )
    d2 = _bilinear_sample_2d(
        dist_m, x=float(c2x) / float(cell_size_m), y=float(c2y) / float(cell_size_m), default=0.0
    )
    thr = float(radius_m) + float(eps_cell_m)
    return (float(d1) <= thr) or (float(d2) <= thr)


def reachable_bicycle_kinematics(
    dist_m: np.ndarray,
    *,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    cell_size_m: float,
    wheelbase_m: float,
    delta_max_rad: float,
    dt: float,
    v_m_s: float,
    primitive_steps: int,
    heading_bins: int,
    radius_m: float,
    x1_m: float,
    x2_m: float,
    eps_cell_m: float,
    goal_tolerance_m: float,
    max_expansions: int,
) -> bool:
    """基于 bicycle 运动学 + 双圆碰撞检测的粗略可达性检查。

    故意设计为轻量级（固定速度 + 恒定转向 primitive），
    仅用于排除在 bicycle 模型下*不可解*的森林地图。
    """
    if int(heading_bins) < 8:
        raise ValueError("heading_bins must be >= 8")
    if int(primitive_steps) < 1:
        raise ValueError("primitive_steps must be >= 1")
    if not (float(cell_size_m) > 0.0):
        raise ValueError("cell_size_m must be > 0")

    h, w = dist_m.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return False

    start_x_m = float(sx) * float(cell_size_m)
    start_y_m = float(sy) * float(cell_size_m)
    goal_x_m = float(gx) * float(cell_size_m)
    goal_y_m = float(gy) * float(cell_size_m)
    psi0 = math.atan2(goal_y_m - start_y_m, goal_x_m - start_x_m)

    if _two_circle_collision(
        dist_m,
        x_m=start_x_m,
        y_m=start_y_m,
        psi_rad=psi0,
        cell_size_m=float(cell_size_m),
        radius_m=float(radius_m),
        x1_m=float(x1_m),
        x2_m=float(x2_m),
        eps_cell_m=float(eps_cell_m),
    ):
        return False

    v = float(max(0.1, float(v_m_s)))
    L = float(wheelbase_m)
    dt_ = float(dt)
    step_dist = v * dt_ * float(primitive_steps)

    def wrap_pi(x: float) -> float:
        return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)

    def key(x_m: float, y_m: float, psi: float) -> tuple[int, int, int]:
        xi = int(round(float(x_m) / float(cell_size_m)))
        yi = int(round(float(y_m) / float(cell_size_m)))
        a = (float(psi) % (2.0 * math.pi)) / (2.0 * math.pi)
        hi = int(round(a * float(heading_bins))) % int(heading_bins)
        return xi, yi, hi

    def h_cost(x_m: float, y_m: float) -> float:
        return float(math.hypot(float(goal_x_m) - float(x_m), float(goal_y_m) - float(y_m)))

    # 转向 primitive（在 delta 空间中大致匹配离散 delta_dot 粒度）。
    dmax = float(delta_max_rad)
    deltas = (-dmax, -(2.0 / 3.0) * dmax, -(1.0 / 3.0) * dmax, 0.0, (1.0 / 3.0) * dmax, (2.0 / 3.0) * dmax, dmax)

    start_state = (start_x_m, start_y_m, wrap_pi(psi0))
    start_key = key(*start_state)

    open_pq: list[tuple[float, float, tuple[float, float, float], tuple[int, int, int]]] = []
    g_best: dict[tuple[int, int, int], float] = {start_key: 0.0}
    open_pq.append((h_cost(start_x_m, start_y_m), 0.0, start_state, start_key))
    import heapq

    heapq.heapify(open_pq)

    expansions = 0
    while open_pq and expansions < int(max_expansions):
        f, g, (x_m, y_m, psi), k = heapq.heappop(open_pq)
        if g != g_best.get(k, float("inf")):
            continue

        if h_cost(x_m, y_m) <= float(goal_tolerance_m):
            return True

        expansions += 1

        for delta in deltas:
            nx, ny, npsi = float(x_m), float(y_m), float(psi)
            ok = True
            for _ in range(int(primitive_steps)):
                nx = float(nx) + v * math.cos(float(npsi)) * dt_
                ny = float(ny) + v * math.sin(float(npsi)) * dt_
                npsi = wrap_pi(float(npsi) + (v / L) * math.tan(float(delta)) * dt_)
                if _two_circle_collision(
                    dist_m,
                    x_m=nx,
                    y_m=ny,
                    psi_rad=npsi,
                    cell_size_m=float(cell_size_m),
                    radius_m=float(radius_m),
                    x1_m=float(x1_m),
                    x2_m=float(x2_m),
                    eps_cell_m=float(eps_cell_m),
                ):
                    ok = False
                    break
            if not ok:
                continue

            nk = key(nx, ny, npsi)
            if not (0 <= nk[0] < w and 0 <= nk[1] < h):
                continue

            ng = float(g) + float(step_dist)
            if ng < g_best.get(nk, float("inf")):
                g_best[nk] = ng
                heapq.heappush(open_pq, (ng + h_cost(nx, ny), ng, (nx, ny, npsi), nk))

    return False


def check_bicycle_reachable(
    dist_m: np.ndarray,
    *,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    cell_size_m: float,
    goal_tolerance_m: float = 0.3,
) -> bool:
    """独立于参评算法的 bicycle-kinematic 可达性检查。

    用于推理阶段筛选随机 (start, goal) pair，替代 Hybrid A*
    以避免 benchmark 公平性偏差。参数使用与环境相同的
    默认运动学常量（wheelbase=0.6m, delta_max=27°, dt=0.05s）。
    """
    eps_cell_m = float(math.sqrt(2.0) * 0.5 * float(cell_size_m))
    # TwoCircleFootprint 默认参数
    r_m = float(math.hypot(0.740 / 2.0, 0.924 / 4.0))
    x1_m = (0.6 / 2.0) - (0.924 / 4.0)
    x2_m = (0.6 / 2.0) + (0.924 / 4.0)
    h, w = dist_m.shape
    return reachable_bicycle_kinematics(
        dist_m,
        start_xy=start_xy,
        goal_xy=goal_xy,
        cell_size_m=float(cell_size_m),
        wheelbase_m=0.6,
        delta_max_rad=float(math.radians(27.0)),
        dt=0.05,
        v_m_s=1.0,
        primitive_steps=4,
        heading_bins=36,
        radius_m=float(r_m),
        x1_m=float(x1_m),
        x2_m=float(x2_m),
        eps_cell_m=float(eps_cell_m),
        goal_tolerance_m=float(goal_tolerance_m),
        max_expansions=int(w * h * 4),
    )


def _reachable_8(free: np.ndarray, start_xy: tuple[int, int], goal_xy: tuple[int, int]) -> bool:
    h, w = free.shape
    sx, sy = start_xy
    gx, gy = goal_xy
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return False
    if not (bool(free[sy, sx]) and bool(free[gy, gx])):
        return False

    q: deque[tuple[int, int]] = deque([(sx, sy)])
    visited = np.zeros((h, w), dtype=np.uint8)
    visited[sy, sx] = 1

    moves = (
        (0, 1),
        (0, -1),
        (-1, 0),
        (1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    while q:
        x, y = q.popleft()
        if x == gx and y == gy:
            return True
        for dx, dy in moves:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if visited[ny, nx]:
                continue
            if not bool(free[ny, nx]):
                continue
            visited[ny, nx] = 1
            q.append((nx, ny))

    return False


def _mark_disk(grid: np.ndarray, *, cx: float, cy: float, r_cells: float) -> None:
    h, w = grid.shape
    x0 = max(0, int(math.floor(cx - r_cells - 1.0)))
    x1 = min(w - 1, int(math.ceil(cx + r_cells + 1.0)))
    y0 = max(0, int(math.floor(cy - r_cells - 1.0)))
    y1 = min(h - 1, int(math.ceil(cy + r_cells + 1.0)))
    if x1 < x0 or y1 < y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= float(r_cells) ** 2
    sub = grid[y0 : y1 + 1, x0 : x1 + 1]
    sub[mask] = 1


def _clear_disk(grid: np.ndarray, *, cx: float, cy: float, r_cells: float) -> None:
    h, w = grid.shape
    x0 = max(0, int(math.floor(cx - r_cells - 1.0)))
    x1 = min(w - 1, int(math.ceil(cx + r_cells + 1.0)))
    y0 = max(0, int(math.floor(cy - r_cells - 1.0)))
    y1 = min(h - 1, int(math.ceil(cy + r_cells + 1.0)))
    if x1 < x0 or y1 < y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= float(r_cells) ** 2
    sub = grid[y0 : y1 + 1, x0 : x1 + 1]
    sub[mask] = 0


def _sample_gap_cells(*, base_gap_m: float, jitter: float, rng: np.random.Generator, cell_size_m: float) -> float:
    base = max(0.0, float(base_gap_m))
    if base <= 0.0:
        return 0.0

    jit = max(0.0, float(jitter))
    scale_jit = rng.uniform(1.0 - jit, 1.0 + jit) if jit > 0.0 else 1.0

    # 混合采样实现"部分宽、部分窄"的通道效果（幅度温和以保持稳定性）。
    mix = rng.choice(np.array([0.85, 1.0, 1.15], dtype=np.float32), p=np.array([0.15, 0.70, 0.15]))
    gap_m = base * float(mix) * float(scale_jit)
    return float(gap_m) / float(cell_size_m)


def _place_trunks(
    *,
    grid: np.ndarray,
    params: ForestParams,
    rng: np.random.Generator,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    footprint_clearance_cells: float,
) -> None:
    h, w = grid.shape
    cell = float(params.cell_size_m)
    boundary_margin = max(0, int(round(float(params.boundary_margin_m) / cell)))

    # 空间哈希用于快速重叠检查（bin 坐标 -> 该 bin 中的树干）。
    r_max_cells = float(params.trunk_radius_m_max) / cell
    gap_max_cells = float(params.trunk_gap_m) * 1.35 * (1.0 + float(params.trunk_gap_jitter)) / cell
    bin_size = max(4.0, (2.0 * r_max_cells) + gap_max_cells)
    bins: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}

    trunks: list[tuple[float, float, float, float]] = []  # cx_cells, cy_cells, r_cells, gap_cells（树干列表）

    tries = 0

    while len(trunks) < int(params.trunk_count) and tries < int(params.trunk_place_tries):
        tries += 1

        r_m = float(rng.uniform(float(params.trunk_radius_m_min), float(params.trunk_radius_m_max)))
        r_cells = r_m / cell
        gap_cells = _sample_gap_cells(
            base_gap_m=float(params.trunk_gap_m),
            jitter=float(params.trunk_gap_jitter),
            rng=rng,
            cell_size_m=cell,
        )
        # 树干间隙独立于车辆轮廓。
        # 森林自然包含紧密聚簇；可行性后续通过 EDT + 可达性检查保证。
        gap_cells = max(0.0, float(gap_cells))

        # 保持树干在边界内，预留小段保护带。
        guard = max(1.0, r_cells + 1.0)
        x_min = float(boundary_margin) + guard
        x_max = float(w - 1 - boundary_margin) - guard
        y_min = float(boundary_margin) + guard
        y_max = float(h - 1 - boundary_margin) - guard
        if x_max <= x_min or y_max <= y_min:
            break

        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))

        # 保持起点/终点区域清空。
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        # 禁入区仅需足够放置初始车辆姿态，避免在起点/终点周围
        # 产生不自然的大片空旷区域。可行性后续通过 EDT + 可达性检查保证。
        keepout = float(footprint_clearance_cells) + r_cells + 2.0
        if (cx - sx) ** 2 + (cy - sy) ** 2 < keepout**2:
            continue
        if (cx - gx) ** 2 + (cy - gy) ** 2 < keepout**2:
            continue

        bx = int(math.floor(cx / bin_size))
        by = int(math.floor(cy / bin_size))

        ok = True
        for nbx in (bx - 1, bx, bx + 1):
            for nby in (by - 1, by, by + 1):
                for ox, oy, or_cells, ogap_cells in bins.get((nbx, nby), []):
                    min_gap = min(float(ogap_cells), float(gap_cells))
                    min_d = float(or_cells) + float(r_cells) + float(min_gap)
                    if (cx - float(ox)) ** 2 + (cy - float(oy)) ** 2 < min_d**2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        trunks.append((cx, cy, r_cells, gap_cells))
        bins.setdefault((bx, by), []).append((cx, cy, r_cells, gap_cells))
        _mark_disk(grid, cx=cx, cy=cy, r_cells=r_cells)

    # 如果放置严重失败，最好在地图层面强制重试。
    if len(trunks) < max(5, int(0.6 * float(params.trunk_count))):
        raise RuntimeError("Trunk placement failed; retry map generation.")


def _place_bushes(
    *,
    grid: np.ndarray,
    params: ForestParams,
    rng: np.random.Generator,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
) -> None:
    if int(params.bush_cluster_count) <= 0:
        return

    h, w = grid.shape
    cell = float(params.cell_size_m)
    boundary_margin = max(0, int(round(float(params.boundary_margin_m) / cell)))

    spread_cells = max(0.1, float(params.bush_spread_m) / cell)

    for _k in range(int(params.bush_cluster_count)):
        cx = float(rng.uniform(boundary_margin + 2, w - 1 - (boundary_margin + 2)))
        cy = float(rng.uniform(boundary_margin + 2, h - 1 - (boundary_margin + 2)))

        # 与起点/终点保持一定距离。
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if (cx - sx) ** 2 + (cy - sy) ** 2 < (8.0 * spread_cells) ** 2:
            continue
        if (cx - gx) ** 2 + (cy - gy) ** 2 < (8.0 * spread_cells) ** 2:
            continue

        n = int(rng.integers(int(params.bush_per_cluster_min), int(params.bush_per_cluster_max) + 1))
        for _j in range(n):
            ox = float(rng.normal(0.0, spread_cells))
            oy = float(rng.normal(0.0, spread_cells))
            bx = float(np.clip(cx + ox, boundary_margin + 1, w - 1 - (boundary_margin + 1)))
            by = float(np.clip(cy + oy, boundary_margin + 1, h - 1 - (boundary_margin + 1)))
            r_m = float(rng.uniform(float(params.bush_radius_m_min), float(params.bush_radius_m_max)))
            _mark_disk(grid, cx=bx, cy=by, r_cells=r_m / cell)


def generate_forest_grid(
    *,
    params: ForestParams,
    rng: np.random.Generator,
    footprint_clearance_m: float,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """生成带有可达性检查的碎片化森林占据栅格。

    Returns:
      - grid (H, W) uint8，y=0 在底部，1=障碍物
      - start_xy, goal_xy（整数栅格坐标）
    """
    w = int(params.width_cells)
    h = int(params.height_cells)
    if w <= 4 or h <= 4:
        raise ValueError("Forest grid is too small.")
    cell = float(params.cell_size_m)
    if not (cell > 0):
        raise ValueError("cell_size_m must be > 0")

    margin_start = max(1, int(round(float(params.start_margin_m) / cell)))
    margin_goal = max(1, int(round(float(params.goal_margin_m) / cell)))

    def clamp_xy(x: int, y: int, margin_cells: int) -> tuple[int, int]:
        return (
            int(np.clip(int(x), margin_cells, w - 1 - margin_cells)),
            int(np.clip(int(y), margin_cells, h - 1 - margin_cells)),
        )

    # 将起点/终点放置在森林内部（远离空白边界）。
    start_xy = clamp_xy(int(round(float(params.start_frac) * float(w - 1))), int(round(float(params.start_frac) * float(h - 1))), margin_start)
    goal_xy = clamp_xy(int(round(float(params.goal_frac) * float(w - 1))), int(round(float(params.goal_frac) * float(h - 1))), margin_goal)

    safe_clearance_cells = float(footprint_clearance_m) / cell

    for _ in range(int(params.max_tries)):
        grid = np.zeros((h, w), dtype=np.uint8)

        try:
            _place_trunks(
                grid=grid,
                params=params,
                rng=rng,
                start_xy=start_xy,
                goal_xy=goal_xy,
                footprint_clearance_cells=safe_clearance_cells,
            )
        except RuntimeError:
            continue

        _place_bushes(grid=grid, params=params, rng=rng, start_xy=start_xy, goal_xy=goal_xy)

        # 清除起点/终点区域的障碍物。
        _clear_disk(grid, cx=float(start_xy[0]), cy=float(start_xy[1]), r_cells=safe_clearance_cells + 1.0)
        _clear_disk(grid, cx=float(goal_xy[0]), cy=float(goal_xy[1]), r_cells=safe_clearance_cells + 1.0)

        # 构型空间近似下的可达性检查：
        # 将 EDT clearance 足够大的状态视为自由。
        grid_top = grid[::-1, :]
        free = (grid_top == 0).astype(np.uint8) * 255
        dist_top = cv2.distanceTransform(
            free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
        ).astype(np.float32)
        dist = dist_top[::-1, :]
        safe_free = dist > safe_clearance_cells

        if not _reachable_8(safe_free, start_xy, goal_xy):
            continue

        # 额外可达性检查：bicycle 模型（最小转弯半径 + 车身轮廓）必须可达。
        dist_m = (dist * float(cell)).astype(np.float32, copy=False)
        eps_cell_m = float(math.sqrt(2.0) * 0.5 * cell)
        # 双圆模型（与 env.py 一致）；从 clearance 输入推导 r 以避免常量漂移。
        r_m = max(0.0, float(footprint_clearance_m) - float(eps_cell_m))
        x1_m = (0.6 / 2.0) - (0.924 / 4.0)
        x2_m = (0.6 / 2.0) + (0.924 / 4.0)

        if not reachable_bicycle_kinematics(
            dist_m,
            start_xy=start_xy,
            goal_xy=goal_xy,
            cell_size_m=float(cell),
            wheelbase_m=0.6,
            delta_max_rad=float(math.radians(27.0)),
            dt=0.05,
            v_m_s=1.0,
            primitive_steps=4,
            heading_bins=36,
            radius_m=float(r_m),
            x1_m=float(x1_m),
            x2_m=float(x2_m),
            eps_cell_m=float(eps_cell_m),
            goal_tolerance_m=0.30,
            max_expansions=int(w * h * 4),
        ):
            continue

        # 通过两项可达性检查。
        return grid.astype(np.uint8, copy=False), start_xy, goal_xy

    raise RuntimeError("Failed to generate a reachable forest map; lower density or increase max_tries.")
