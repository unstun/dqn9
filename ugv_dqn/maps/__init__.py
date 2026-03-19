"""所有支持环境的地图定义与加载。

地图类型
---------
- 森林地图（程序化生成）：forest_a / forest_b / forest_c / forest_d
  通过 forest.py 中的 ForestParams + generate_forest_grid() 生成。
  首次访问后缓存于 _FOREST_CACHE。

- 真实世界地图（PGM）：realmap_a
  通过 pgm.py 从 realmap/map_a.pgm 加载。
  起点/终点由 EDT clearance 分析选定。

主要导出
-----------
- MapSpec protocol        接口：name, start_xy, goal_xy, size, obstacle_grid()。
- GridMapSpec             字符串行地图规格（旧版栅格地图）。
- ArrayGridMapSpec        Numpy 数组地图规格（森林 + 真实地图）。
- get_map_spec(name)      统一入口：根据环境名返回 MapSpec。
- FOREST_ENV_ORDER        ("forest_a", "forest_b", "forest_c", "forest_d")
- REALMAP_ENV_ORDER       ("realmap_a",)
- ALL_ENV_ORDER           FOREST + REALMAP 合集。

预计算专家路径
------------------------
maps/precomputed/*.json   Hybrid A* 参考路径（供 train.py 中 DQfD 使用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MapSpec(Protocol):
    name: str
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]: ...

    def obstacle_grid(self) -> np.ndarray: ...


@dataclass(frozen=True)
class GridMapSpec:
    name: str
    rows_y0_bottom: list[str]
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        height = len(self.rows_y0_bottom)
        width = len(self.rows_y0_bottom[0]) if height else 0
        return width, height

    def obstacle_grid(self) -> np.ndarray:
        """返回 (H, W) uint8 数组，y=0 在底部，1=障碍物。"""
        width, height = self.size
        if height == 0 or width == 0:
            raise ValueError(f"Empty map: {self.name!r}")
        if any(len(r) != width for r in self.rows_y0_bottom):
            raise ValueError(f"Non-rectangular map: {self.name!r}")

        grid = np.zeros((height, width), dtype=np.uint8)
        for y, row in enumerate(self.rows_y0_bottom):
            for x, ch in enumerate(row):
                if ch == "#":
                    grid[y, x] = 1
                elif ch == ".":
                    continue
                else:
                    raise ValueError(
                        f"Invalid char {ch!r} in map {self.name!r} at (x={x}, y={y})"
                    )
        return grid


@dataclass(frozen=True)
class ArrayGridMapSpec:
    name: str
    grid_y0_bottom: np.ndarray  # (H, W) uint8，y=0 在底部，1=障碍物
    start_xy: tuple[int, int]
    goal_xy: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.grid_y0_bottom.shape
        return int(w), int(h)

    def obstacle_grid(self) -> np.ndarray:
        return self.grid_y0_bottom.astype(np.uint8, copy=True)


# 旧版栅格地图（"a"..."d"）已移除；保留符号以向后兼容。
ENV_ORDER: tuple[str, ...] = ()
FOREST_ENV_ORDER: tuple[str, ...] = ("forest_a", "forest_b", "forest_c", "forest_d")

# 真实世界 PGM 地图（按森林类型处理：使用相同参数的 UGVBicycleEnv）
REALMAP_ENV_ORDER: tuple[str, ...] = ("realmap_a",)

# 对外使用的组合环境顺序
ALL_ENV_ORDER: tuple[str, ...] = FOREST_ENV_ORDER + REALMAP_ENV_ORDER


MAPS: dict[str, GridMapSpec] = {}


_FOREST_CACHE: dict[str, ArrayGridMapSpec] = {}
_REALMAP_CACHE: dict[str, ArrayGridMapSpec] = {}


def _get_realmap_spec(env_name: str) -> ArrayGridMapSpec:
    """加载真实世界 PGM 地图为 ArrayGridMapSpec（y=0 在底部，1=障碍物）。"""
    if env_name in _REALMAP_CACHE:
        return _REALMAP_CACHE[env_name]
    from ugv_dqn.maps.pgm import load_pgm_map
    if env_name == "realmap_a":
        pgm_path = str(_PROJECT_ROOT / "realmap" / "map_a.pgm")
        # 通过距离变换分析找到的最佳起点/终点（各自 clearance >1.8m）
        start_xy = (34, 29)   # x=34, y=29 (y=0 at bottom), clearance≈1.88m
        goal_xy  = (371, 109) # x=371, y=109, clearance≈2.20m
    else:
        raise KeyError(env_name)
    spec = load_pgm_map(pgm_path, start_xy, goal_xy, name=env_name)
    _REALMAP_CACHE[env_name] = spec
    return spec


def _get_forest_spec(env_name: str) -> ArrayGridMapSpec:
    if env_name in _FOREST_CACHE:
        return _FOREST_CACHE[env_name]

    from ugv_dqn.maps.forest import ForestParams, generate_forest_grid

    if env_name == "forest_a":
        # 最大森林地图（空间更大；更易观察泛化效果）。
        seed = 101
        params = ForestParams(
            width_cells=360,
            height_cells=360,
            trunk_count=85,
            trunk_gap_m=3.0,
            trunk_gap_jitter=0.15,
            bush_cluster_count=0,
            start_frac=0.12,
            goal_frac=0.88,
        )
    elif env_name == "forest_b":
        # 小地图，间隙更窄（须保持 bicycle 运动学可行性）。
        seed = 202
        params = ForestParams(
            width_cells=96,
            height_cells=96,
            trunk_count=28,
            trunk_gap_m=1.35,
            bush_cluster_count=0,
        )
    elif env_name == "forest_c":
        # 大地图，更密集布局（须保持 bicycle 运动学可行性）。
        seed = 303
        params = ForestParams(
            width_cells=160,
            height_cells=160,
            trunk_count=85,
            trunk_gap_m=1.25,
            bush_cluster_count=0,
        )
    elif env_name == "forest_d":
        # 小地图，间隙较宽。
        seed = 404
        params = ForestParams(
            width_cells=96,
            height_cells=96,
            trunk_count=28,
            trunk_gap_m=1.30,
            bush_cluster_count=0,
        )
    else:
        raise KeyError(env_name)

    # 可达性检查的车身 clearance：(r + safe_distance + eps_cell)。
    # 确保生成的森林地图不仅无碰撞，还具备足够的 clearance
    # 以满足奖励函数的安全距离阈值。
    # r=0.436m 对应双圆近似，eps_cell=sqrt(2)/2*cell_size。
    r_m = 0.436
    eps_cell_m = (2.0**0.5) * 0.5 * float(params.cell_size_m)
    safe_distance_m = 0.20
    footprint_clearance_m = r_m + safe_distance_m + eps_cell_m

    grid, start_xy, goal_xy = generate_forest_grid(
        params=params,
        rng=np.random.default_rng(seed),
        footprint_clearance_m=footprint_clearance_m,
    )
    spec = ArrayGridMapSpec(name=env_name, grid_y0_bottom=grid, start_xy=start_xy, goal_xy=goal_xy)
    _FOREST_CACHE[env_name] = spec
    return spec


def get_map_spec(env_name: str) -> MapSpec:
    if env_name in REALMAP_ENV_ORDER:
        return _get_realmap_spec(env_name)
    if env_name in FOREST_ENV_ORDER:
        return _get_forest_spec(env_name)
    try:
        return MAPS[env_name]
    except KeyError as e:
        raise KeyError(
            f"Unknown env {env_name!r}. Options: {sorted(list(MAPS) + list(FOREST_ENV_ORDER) + list(REALMAP_ENV_ORDER))}"
        ) from e
