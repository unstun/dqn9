"""DQN 智能体的 Q 网络架构。

提供
--------
- MLPQNetwork        全连接 Q 网络（可变深度/宽度）。
- CNNQNetwork        CNN 双输入 Q 网络（消融实验推荐，默认架构）。
                     采用标量特征 + 2D 地图通道的双分支输入：
                     卷积分支处理地图通道，全连接分支处理标量，
                     两路特征拼接后送入 MLP 头部输出 Q 值。
                     消融实验表明 CNN 双输入显著优于纯 MLP（仅标量拼接
                     展平地图），在森林环境中成功率和路径质量均更高。
- infer_flat_obs_cnn_layout()
                     从 obs_dim 自动推断 (scalar_dim, map_channels, map_size)。

观测格式（扁平向量）
--------------------------------
UGVBicycleEnv: [11 个标量] + [3 x N^2 地图]      -> obs_dim = 11 + 3*N^2
               (地图 = occupancy, goal distance, EDT clearance)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from ugv_dqn.modules import (
    CoordAttention,
    DeformConv2dBlock,
    FADC,
    IQNHead,
    NoisyLinear,
    SpatialMHA,
)


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, hidden_layers: int = 2, **_kw: object):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 向后兼容的名称（历史上本仓库只有 MLP Q 网络）。
QNetwork = MLPQNetwork


@dataclass(frozen=True)
class FlatObsCnnLayout:
    scalar_dim: int
    map_channels: int
    map_size: int


def infer_flat_obs_cnn_layout(obs_dim: int) -> FlatObsCnnLayout:
    """根据本仓库的扁平观测推断 (scalar_dim, map_channels, map_size)。

    支持的布局：
    - UGVBicycleEnv:obs = [11 个标量] + [3 * (N*N) 地图]  (occ + cost + edt)
    """

    d = int(obs_dim)
    if d <= 0:
        raise ValueError("obs_dim must be > 0")

    candidates: list[FlatObsCnnLayout] = []
    for scalar_dim, channels in ((5, 1), (11, 3)):
        rem = d - int(scalar_dim)
        if rem <= 0:
            continue
        if rem % int(channels) != 0:
            continue
        per = rem // int(channels)
        n = int(round(math.sqrt(per)))
        if n > 0 and n * n == per:
            candidates.append(FlatObsCnnLayout(scalar_dim=int(scalar_dim), map_channels=int(channels), map_size=int(n)))

    if not candidates:
        raise ValueError(
            f"Cannot infer CNN layout from obs_dim={d}. Expected 11+3*N^2 (UGVBicycleEnv) or 5+N^2 (legacy)."
        )
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous CNN layout for obs_dim={d}: {candidates}")
    return candidates[0]


def _make_linear(in_f: int, out_f: int, *, noisy: bool) -> nn.Module:
    """创建 Linear 或 NoisyLinear 层。"""
    if noisy:
        return NoisyLinear(in_f, out_f)
    return nn.Linear(in_f, out_f)


class CNNQNetwork(nn.Module):
    """CNN 双输入 Q 网络（消融实验证明优于纯 MLP 架构）。

    双分支结构：
      1. 卷积分支：将 (map_channels, map_size, map_size) 地图通道送入 Conv2D 骨干提取空间特征
      2. 标量分支：11 维标量（位姿、速度、转向角、goal distance 等）
    两路特征拼接后经 MLP 头部输出各动作的 Q 值。

    消融实验结论：CNN 双输入 vs 纯 MLP
      - CNN 能有效利用占据栅格、goal distance field、EDT 安全距离的空间结构
      - 纯 MLP 将地图展平后丧失空间相邻关系，收敛慢且泛化差
      - 在森林环境成功率和路径质量指标上 CNN 双输入均显著优于 MLP
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        scalar_dim: int,
        map_channels: int,
        map_size: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        dueling: bool = False,
        mha: bool = False,
        mha_heads: int = 4,
        coord_attn: bool = False,
        noisy: bool = False,
        fadc: bool = False,
        deform: bool = False,
        iqn: bool = False,
        iqn_cos: int = 64,
        iqn_quantiles: int = 8,
    ) -> None:
        super().__init__()

        self.scalar_dim = int(scalar_dim)
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dueling = bool(dueling)
        self.use_iqn = bool(iqn)
        self.use_noisy = bool(noisy)

        if self.scalar_dim < 0:
            raise ValueError("scalar_dim must be >= 0")
        if self.map_channels < 1:
            raise ValueError("map_channels must be >= 1")
        if self.map_size < 1:
            raise ValueError("map_size must be >= 1")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        expected = int(self.scalar_dim) + int(self.map_channels) * int(self.map_size) * int(self.map_size)
        if int(input_dim) != expected:
            raise ValueError(
                f"CNNQNetwork expected input_dim={expected} (scalar_dim={self.scalar_dim}, "
                f"map_channels={self.map_channels}, map_size={self.map_size}), got {int(input_dim)}"
            )

        # ---------- 卷积骨干网络 ----------
        if fadc:
            # 用 FADC 替换中间卷积层
            self.conv = nn.Sequential(
                nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                FADC(32, 64, kernel_size=3, stride=2, padding=1, max_dilation=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        elif deform:
            # 用 DeformConv2d 替换中间卷积层
            self.conv = nn.Sequential(
                nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                DeformConv2dBlock(32, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

        # ---------- 卷积后注意力 ----------
        self.spatial_mha: SpatialMHA | None = SpatialMHA(64, mha_heads) if mha else None
        self.coord_attn: CoordAttention | None = CoordAttention(64) if coord_attn else None

        with torch.no_grad():
            dummy = torch.zeros((1, self.map_channels, self.map_size, self.map_size), dtype=torch.float32)
            conv_out = self.conv(dummy)
            if self.spatial_mha is not None:
                conv_out = self.spatial_mha(conv_out)
            if self.coord_attn is not None:
                conv_out = self.coord_attn(conv_out)
            conv_out_dim = int(conv_out.flatten(start_dim=1).shape[1])
        fc_in_dim = int(self.scalar_dim) + int(conv_out_dim)

        # ---------- IQN 头部 ----------
        self.iqn_head: IQNHead | None = None
        self.iqn_value_head: IQNHead | None = None
        self.iqn_advantage_head: IQNHead | None = None
        if self.use_iqn:
            # 特征提取主干（无最终 Q 层 — 由 IQN 负责）
            trunk: list[nn.Module] = []
            trunk.append(_make_linear(fc_in_dim, int(hidden_dim), noisy=False))
            trunk.append(nn.ReLU())
            for _ in range(max(0, int(hidden_layers) - 1)):
                trunk.append(_make_linear(int(hidden_dim), int(hidden_dim), noisy=False))
                trunk.append(nn.ReLU())
            self.trunk = nn.Sequential(*trunk)

            if self.dueling:
                # IQN + Dueling：Value 和 Advantage 各用独立 IQN 头
                self.iqn_value_head = IQNHead(int(hidden_dim), 1, n_cos=iqn_cos, n_quantiles=iqn_quantiles)
                self.iqn_advantage_head = IQNHead(int(hidden_dim), int(output_dim), n_cos=iqn_cos, n_quantiles=iqn_quantiles)
            else:
                self.iqn_head = IQNHead(int(hidden_dim), int(output_dim), n_cos=iqn_cos, n_quantiles=iqn_quantiles)
            self.head = None  # type: ignore[assignment]
            # 不 return — 下方 elif/else 靠条件跳过

        # ---------- Dueling / 标准全连接头部 ----------
        # NoisyNet 优化：仅在输出头部添加噪声，共享主干不加
        # （与原论文一致 — 探索噪声仅在决策层）
        elif self.dueling:
            shared: list[nn.Module] = []
            shared.append(_make_linear(fc_in_dim, int(hidden_dim), noisy=False))
            shared.append(nn.ReLU())
            for _ in range(max(0, int(hidden_layers) - 2)):
                shared.append(_make_linear(int(hidden_dim), int(hidden_dim), noisy=False))
                shared.append(nn.ReLU())
            self.shared = nn.Sequential(*shared)
            self.value_stream = nn.Sequential(
                _make_linear(int(hidden_dim), int(hidden_dim), noisy=noisy),
                nn.ReLU(),
                _make_linear(int(hidden_dim), 1, noisy=noisy),
            )
            self.advantage_stream = nn.Sequential(
                _make_linear(int(hidden_dim), int(hidden_dim), noisy=noisy),
                nn.ReLU(),
                _make_linear(int(hidden_dim), int(output_dim), noisy=noisy),
            )
            self.head = None  # type: ignore[assignment]
        else:
            # 标准头部：主干层为普通层，仅最后一层添加噪声
            layers: list[nn.Module] = []
            layers.append(_make_linear(fc_in_dim, int(hidden_dim), noisy=False))
            layers.append(nn.ReLU())
            for _ in range(int(hidden_layers) - 1):
                layers.append(_make_linear(int(hidden_dim), int(hidden_dim), noisy=False))
                layers.append(nn.ReLU())
            layers.append(_make_linear(int(hidden_dim), int(output_dim), noisy=noisy))
            self.head = nn.Sequential(*layers)

    def forward_quantiles(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """返回逐分位数 Q 值 (B, K, n_actions)。仅 IQN 模式可用。"""
        if not self.use_iqn:
            raise RuntimeError("forward_quantiles requires iqn=True")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        scalars = x[:, : self.scalar_dim]
        maps_flat = x[:, self.scalar_dim :]
        maps = maps_flat.reshape(x.shape[0], self.map_channels, self.map_size, self.map_size)
        conv = self.conv(maps)
        if self.spatial_mha is not None:
            conv = self.spatial_mha(conv)
        if self.coord_attn is not None:
            conv = self.coord_attn(conv)
        feats = torch.cat([scalars, conv.flatten(start_dim=1)], dim=1)
        trunk_out = self.trunk(feats)

        if self.dueling and self.iqn_value_head is not None and self.iqn_advantage_head is not None:
            v_q = self.iqn_value_head.forward_quantiles(trunk_out, tau)       # (B, K, 1)
            a_q = self.iqn_advantage_head.forward_quantiles(trunk_out, tau)   # (B, K, n_actions)
            return v_q + a_q - a_q.mean(dim=2, keepdim=True)                 # (B, K, n_actions)
        assert self.iqn_head is not None
        return self.iqn_head.forward_quantiles(trunk_out, tau)                # (B, K, n_actions)

    def reset_noise(self) -> None:
        """重置所有 NoisyLinear 层的噪声。"""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("CNNQNetwork expects (batch, obs_dim) input")
        if int(x.shape[1]) != int(self.input_dim):
            raise ValueError(f"CNNQNetwork expected input_dim={self.input_dim}, got {int(x.shape[1])}")

        scalars = x[:, : self.scalar_dim]
        maps_flat = x[:, self.scalar_dim :]
        maps = maps_flat.reshape(int(x.shape[0]), self.map_channels, self.map_size, self.map_size)
        conv = self.conv(maps)                       # (B, 64, H', W')

        if self.spatial_mha is not None:
            conv = self.spatial_mha(conv)
        if self.coord_attn is not None:
            conv = self.coord_attn(conv)

        conv_flat = conv.flatten(start_dim=1)
        feats = torch.cat([scalars, conv_flat], dim=1)

        if self.use_iqn:
            trunk_out = self.trunk(feats)
            if self.dueling and self.iqn_value_head is not None and self.iqn_advantage_head is not None:
                v = self.iqn_value_head(trunk_out)          # (B, 1)
                a = self.iqn_advantage_head(trunk_out)      # (B, n_actions)
                return v + a - a.mean(dim=1, keepdim=True)
            assert self.iqn_head is not None
            return self.iqn_head(trunk_out)

        if self.dueling:
            shared_out = self.shared(feats)
            value = self.value_stream(shared_out)            # (B, 1)
            advantage = self.advantage_stream(shared_out)    # (B, n_actions)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

        return self.head(feats)
