"""在线奖励归一化（Welford 在线均值/标准差）。

V8→V9：在采样时进行归一化，而非在收集时。
clip 作为归一化值的安全上下限。
"""

from __future__ import annotations

import numpy as np
import torch


class RunningRewardNormalizer:
    """基于 Welford 在线算法的运行均值/标准差，用于奖励归一化。"""

    def __init__(self, clip: float = 5.0, eps: float = 1e-8):
        self.clip = float(clip)
        self.eps = float(eps)
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return max(float(np.sqrt(self._M2 / self._count)), self.eps)

    def update(self, reward: float) -> None:
        """更新运行统计量（对原始奖励调用，包括示范数据）。"""
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        delta2 = reward - self._mean
        self._M2 += delta * delta2

    def normalize(self, reward: float) -> float:
        """归一化 + 裁剪。需先调用 update()。"""
        if self._count < 2:
            return float(np.clip(reward, -self.clip, self.clip))
        normed = (reward - self._mean) / self.std
        return float(np.clip(normed, -self.clip, self.clip))

    def normalize_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        """向量化归一化 + 裁剪，用于一批奖励（不更新统计量）。"""
        if self._count < 2:
            return rewards.clamp(-self.clip, self.clip)
        normed = (rewards - self._mean) / self.std
        return normed.clamp(-self.clip, self.clip)
