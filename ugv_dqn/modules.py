"""CNN Q 网络的可插拔模块。

提供
--------
- SpatialMHA        对 CNN 特征图空间位置的多头自注意力。
- CoordAttention    坐标注意力 (Hou et al., CVPR 2021)。
- NoisyLinear       分解高斯噪声线性层 (Fortunato et al., ICLR 2018)。
- FADC              频率自适应空洞卷积 (Chen et al., CVPR 2024)。
- DeformConv2dBlock 可变形卷积 v2 封装（通过 torchvision）。
- IQNHead           IQN 余弦嵌入头部 (Dabney et al., ICML 2018)。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# SpatialMHA（空间多头注意力）
# ---------------------------------------------------------------------------

class SpatialMHA(nn.Module):
    """对特征图空间位置的多头自注意力。

    将每个空间位置 (H*W) 视为具有 *channels* 维度的 token。
    应用标准多头注意力，然后进行残差连接和 LayerNorm。
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)       # (B, H*W, C)
        out, _ = self.mha(tokens, tokens, tokens)    # 自注意力
        out = self.norm(tokens + out)                # 残差 + LN
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# 坐标注意力 (Hou et al., CVPR 2021)
# ---------------------------------------------------------------------------

class CoordAttention(nn.Module):
    """坐标注意力：将通道注意力分解为两个一维方向编码（H 和 W），
    保留位置信息。

    参考文献：Hou et al., "Coordinate Attention for Efficient Mobile
    Network Design", CVPR 2021。
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc_shared = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.fc_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.fc_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 沿 W 方向池化 → (B, C, H, 1)，沿 H 方向池化 → (B, C, 1, W)
        x_h = x.mean(dim=3, keepdim=True)           # (B, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True)            # (B, C, 1, W)
        # 沿空间维度拼接，用于共享变换
        x_w_perm = x_w.permute(0, 1, 3, 2)           # (B, C, W, 1)
        cat = torch.cat([x_h, x_w_perm], dim=2)      # (B, C, H+W, 1)
        cat = self.fc_shared(cat)                     # (B, mid, H+W, 1)
        a_h, a_w = torch.split(cat, [H, W], dim=2)
        a_h = self.fc_h(a_h).sigmoid()               # (B, C, H, 1)
        a_w = self.fc_w(a_w.permute(0, 1, 3, 2)).sigmoid()  # (B, C, 1, W)
        return x * a_h * a_w


# ---------------------------------------------------------------------------
# NoisyLinear（噪声线性层，Fortunato et al., ICLR 2018）
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """分解高斯噪声线性层。

    替代标准 nn.Linear；注入可学习噪声以促进探索。
    参考文献：Fortunato et al., "Noisy Networks for Exploration", ICLR 2018。
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _factorised_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._factorised_noise(self.in_features)
        eps_out = self._factorised_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# FADC — 频率自适应空洞卷积 (Chen et al., CVPR 2024)
# 简化版本：可学习的逐通道膨胀率 + 自适应卷积核。
# ---------------------------------------------------------------------------

class FADC(nn.Module):
    """频率自适应空洞卷积（简化版）。

    学习逐通道的软膨胀率，并通过插值空洞卷积来应用。
    可作为 Conv2d 的即插即用替代。

    参考文献：Chen et al., "Frequency-Adaptive Dilated Convolution for
    Semantic Segmentation", CVPR 2024 (Highlight)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        max_dilation: int = 3,
    ) -> None:
        super().__init__()
        self.max_dilation = max_dilation
        # 标准卷积 (dilation=1)
        self.conv_base = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        # 空洞卷积 (dilation=max_dilation)
        pad_d = (kernel_size + (kernel_size - 1) * (max_dilation - 1)) // 2
        self.conv_dilated = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad_d, dilation=max_dilation, bias=False)
        # 可学习混合系数 (sigmoid → [0,1])
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.alpha)
        out = (1.0 - a) * self.conv_base(x) + a * self.conv_dilated(x)
        return F.relu(self.bn(out), inplace=True)


# ---------------------------------------------------------------------------
# DeformConv2dBlock — 可变形卷积封装（通过 torchvision）
# ---------------------------------------------------------------------------

class DeformConv2dBlock(nn.Module):
    """使用 torchvision.ops.deform_conv2d 的可变形卷积 v2 模块。

    为每个卷积核位置学习空间偏移（和调制掩码），
    使感受野能够自适应输入的几何形状。

    若 torchvision 不可用则回退到标准 Conv2d。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # 偏移量：每个位置 2 * kH * kW 个值
        # 掩码：每个位置 kH * kW 个值
        n_offset = 2 * kernel_size * kernel_size
        n_mask = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, n_offset + n_mask, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)  # type: ignore[arg-type]

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        # 在初始化时检查 torchvision 是否可用
        try:
            from torchvision.ops import deform_conv2d as _dcn  # noqa: F401
            self._has_dcn = True
        except ImportError:
            self._has_dcn = False
            self._fallback = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._has_dcn:
            return F.relu(self.bn(self._fallback(x)), inplace=True)

        from torchvision.ops import deform_conv2d

        om = self.offset_conv(x)
        n_offset = 2 * self.kernel_size * self.kernel_size
        offset = om[:, :n_offset, :, :]
        mask = torch.sigmoid(om[:, n_offset:, :, :])
        out = deform_conv2d(x, offset, self.weight, self.bias,
                            stride=(self.stride, self.stride),
                            padding=(self.padding, self.padding),
                            mask=mask)
        return F.relu(self.bn(out), inplace=True)


# ---------------------------------------------------------------------------
# IQN 头部 — 隐式分位数网络 (Dabney et al., ICML 2018)
# ---------------------------------------------------------------------------

class IQNHead(nn.Module):
    """IQN 余弦嵌入头部。

    将特征向量 z 和采样的分位数分数 τ ∈ (0,1) 映射为
    逐动作的分位数值。最终 Q(s,a) 为 K 个样本的均值。

    参考文献：Dabney et al., "Implicit Quantile Networks for Distributional
    Reinforcement Learning", ICML 2018。
    """

    def __init__(self, feature_dim: int, n_actions: int, n_cos: int = 64, n_quantiles: int = 8) -> None:
        super().__init__()
        self.n_cos = n_cos
        self.n_quantiles = n_quantiles
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        # 余弦嵌入：τ → cos(i π τ)，i=0..n_cos-1 → 线性层 → feature_dim
        self.cos_embedding = nn.Linear(n_cos, feature_dim)
        # 乘积后隐藏层 + Q 值输出层（论文 Eq.4: f = FC+ReLU+FC）
        self.fc_hidden = nn.Linear(feature_dim, feature_dim)
        self.q_layer = nn.Linear(feature_dim, n_actions)

    def forward(self, features: torch.Tensor, n_quantiles: int | None = None) -> torch.Tensor:
        """通过对采样分位数取均值，返回平均 Q 值 (B, n_actions)。"""
        K = n_quantiles or self.n_quantiles
        B = features.shape[0]

        # 采样 τ ~ U(0,1): (B, K)
        if self.training:
            tau = torch.rand(B, K, device=features.device, dtype=features.dtype)
        else:
            # 评估时使用确定性分位数
            tau = torch.linspace(0.5 / K, 1.0 - 0.5 / K, K, device=features.device, dtype=features.dtype)
            tau = tau.unsqueeze(0).expand(B, -1)

        # 余弦基：(B, K, n_cos)，i=0..n_cos-1（论文 Eq.4，含 cos(0)=1 常数基）
        i_pi = math.pi * torch.arange(0, self.n_cos, device=features.device, dtype=features.dtype)
        cos_features = torch.cos(tau.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0))  # (B, K, n_cos)
        tau_embed = F.relu(self.cos_embedding(cos_features))  # (B, K, feature_dim)

        # 逐元素相乘 + 隐藏层（论文: Z_τ = f(ψ(x) ⊙ φ(τ)), f = FC+ReLU+FC）
        combined = features.unsqueeze(1) * tau_embed  # (B, K, feature_dim)
        hidden = F.relu(self.fc_hidden(combined))      # (B, K, feature_dim)
        q_quantiles = self.q_layer(hidden)             # (B, K, n_actions)

        return q_quantiles.mean(dim=1)  # (B, n_actions)

    def forward_quantiles(self, features: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """给定显式 τ (B, K)，返回分位数 Q 值 (B, K, n_actions)。"""
        i_pi = math.pi * torch.arange(0, self.n_cos, device=features.device, dtype=features.dtype)
        cos_features = torch.cos(tau.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0))
        tau_embed = F.relu(self.cos_embedding(cos_features))
        combined = features.unsqueeze(1) * tau_embed
        hidden = F.relu(self.fc_hidden(combined))
        return self.q_layer(hidden)
