"""加载 ROS 规范的 PGM/YAML 占据地图到 ArrayGridMapSpec。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ugv_dqn.maps import ArrayGridMapSpec


def load_pgm_map(
    pgm_path: str | Path,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    *,
    name: str = "pgm_map",
    occupied_thresh: float = 0.65,
    negate: bool = False,
) -> ArrayGridMapSpec:
    """加载 ROS 规范的 PGM 占据地图到 ArrayGridMapSpec。

    ROS 规范（negate=False）：
      - 像素越亮 → 越自由（p = pixel/255，低 p = 占据）
      - pixel==0   → p=0.0 < free_thresh  → 占据
      - pixel==254 → p≈1.0 > free_thresh  → 自由

    转换为栅格（y=0 在底部，1=障碍物）：
      当 (1 - pixel/255) > occupied_thresh 时判为占据（即 pixel < (1-thresh)*255）

    Args:
        pgm_path: .pgm 文件路径
        start_xy: 栅格坐标 (x, y)（y=0 在底部）
        goal_xy:  栅格坐标 (x, y)（y=0 在底部）
        name:     地图标识符
        occupied_thresh: .yaml 中的 ROS occupied_thresh（默认 0.65）
        negate:   ROS negate 标志（默认 False）
    """
    pgm_path = Path(pgm_path)
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load PGM: {pgm_path}")

    H, W = img.shape
    # 计算占据概率
    p = img.astype(np.float32) / 255.0
    if negate:
        p = 1.0 - p

    # ROS：占据概率 = 1 - p（越亮 = 越自由）
    occ_prob = 1.0 - p
    obstacle = (occ_prob > float(occupied_thresh)).astype(np.uint8)

    # 垂直翻转：图像第0行在顶部；栅格 y=0 在底部
    grid_y0_bottom = np.flipud(obstacle).copy()

    return ArrayGridMapSpec(
        name=name,
        grid_y0_bottom=grid_y0_bottom,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
