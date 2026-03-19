"""基于 Chaikin 角切割的路径平滑。

对每个线段 (p0, p1) 反复用沿线段 25%/75% 处的两个新点替换，
保留起点和终点。用于在 KPI 评估前对 DQN 路径和经典规划器
输出进行后处理。
"""

from __future__ import annotations

import numpy as np


def chaikin_smooth(points_xy: np.ndarray, *, iterations: int = 2) -> np.ndarray:
    """Chaikin 角切割算法，用于折线平滑。

    Args:
        points_xy: (N, 2) 数组。
        iterations: 细化迭代次数。

    Returns:
        (M, 2) 平滑后的点数组。
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if len(pts) < 3 or iterations <= 0:
        return pts

    out = pts
    for _ in range(int(iterations)):
        new_pts = [out[0]]
        for p0, p1 in zip(out[:-1], out[1:], strict=False):
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts.append(out[-1])
        out = np.vstack(new_pts).astype(np.float32, copy=False)
    return out

