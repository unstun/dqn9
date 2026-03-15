"""Plug-in modules for CNN Q-networks.

Provides
--------
- SpatialMHA        Multi-head self-attention over spatial positions of a CNN feature map.
- CoordAttention    Coordinate Attention (Hou et al., CVPR 2021).
- NoisyLinear       Factorised Gaussian noisy linear layer (Fortunato et al., ICLR 2018).
- FADC              Frequency-Adaptive Dilated Convolution (Chen et al., CVPR 2024).
- DeformConv2dBlock Deformable Convolution v2 wrapper (via torchvision).
- IQNHead           Implicit Quantile Network cosine-embedding head (Dabney et al., ICML 2018).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# SpatialMHA (existing)
# ---------------------------------------------------------------------------

class SpatialMHA(nn.Module):
    """Multi-head self-attention over spatial positions of a feature map.

    Treats each spatial position (H*W) as a token with *channels* dimensions.
    Applies standard multi-head attention followed by a residual connection
    and LayerNorm.
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
        out, _ = self.mha(tokens, tokens, tokens)    # self-attention
        out = self.norm(tokens + out)                # residual + LN
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Coordinate Attention (Hou et al., CVPR 2021)
# ---------------------------------------------------------------------------

class CoordAttention(nn.Module):
    """Coordinate Attention: factorises channel attention into two 1-D
    directional encodings (H and W), preserving positional information.

    Reference: Hou et al., "Coordinate Attention for Efficient Mobile
    Network Design", CVPR 2021.
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
        # Pool along W → (B, C, H, 1), pool along H → (B, C, 1, W)
        x_h = x.mean(dim=3, keepdim=True)           # (B, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True)            # (B, C, 1, W)
        # Concatenate along spatial dim for shared transform
        x_w_perm = x_w.permute(0, 1, 3, 2)           # (B, C, W, 1)
        cat = torch.cat([x_h, x_w_perm], dim=2)      # (B, C, H+W, 1)
        cat = self.fc_shared(cat)                     # (B, mid, H+W, 1)
        a_h, a_w = torch.split(cat, [H, W], dim=2)
        a_h = self.fc_h(a_h).sigmoid()               # (B, C, H, 1)
        a_w = self.fc_w(a_w.permute(0, 1, 3, 2)).sigmoid()  # (B, C, 1, W)
        return x * a_h * a_w


# ---------------------------------------------------------------------------
# NoisyLinear (Fortunato et al., ICLR 2018)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer.

    Replaces standard nn.Linear; injects learnable noise for exploration.
    Reference: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018.
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
# FADC — Frequency-Adaptive Dilated Convolution (Chen et al., CVPR 2024)
# Simplified version: learnable per-channel dilation rate + adaptive kernel.
# ---------------------------------------------------------------------------

class FADC(nn.Module):
    """Frequency-Adaptive Dilated Convolution (simplified).

    Learns a per-channel soft dilation rate and applies it via interpolated
    dilated convolutions. Suitable as a drop-in replacement for Conv2d.

    Reference: Chen et al., "Frequency-Adaptive Dilated Convolution for
    Semantic Segmentation", CVPR 2024 (Highlight).
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
        # Standard conv (dilation=1)
        self.conv_base = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        # Dilated conv (dilation=max_dilation)
        pad_d = (kernel_size + (kernel_size - 1) * (max_dilation - 1)) // 2
        self.conv_dilated = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad_d, dilation=max_dilation, bias=False)
        # Learnable mixing coefficient (sigmoid → [0,1])
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.alpha)
        out = (1.0 - a) * self.conv_base(x) + a * self.conv_dilated(x)
        return F.relu(self.bn(out), inplace=True)


# ---------------------------------------------------------------------------
# DeformConv2dBlock — Deformable Convolution wrapper (via torchvision)
# ---------------------------------------------------------------------------

class DeformConv2dBlock(nn.Module):
    """Deformable Convolution v2 block using torchvision.ops.deform_conv2d.

    Learns spatial offsets (and modulation masks) for each kernel position,
    allowing the receptive field to adapt to input geometry.

    Falls back to standard Conv2d if torchvision is unavailable.
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

        # Offset: 2 * kH * kW values per position
        # Mask: kH * kW values per position
        n_offset = 2 * kernel_size * kernel_size
        n_mask = kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(in_channels, n_offset + n_mask, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)  # type: ignore[arg-type]

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        # Check torchvision availability at init time
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
# IQN Head — Implicit Quantile Network (Dabney et al., ICML 2018)
# ---------------------------------------------------------------------------

class IQNHead(nn.Module):
    """Implicit Quantile Network cosine-embedding head.

    Maps a feature vector z and sampled quantile fractions τ ∈ (0,1) to
    per-action quantile values. The final Q(s,a) is the mean over K samples.

    Reference: Dabney et al., "Implicit Quantile Networks for Distributional
    Reinforcement Learning", ICML 2018.
    """

    def __init__(self, feature_dim: int, n_actions: int, n_cos: int = 64, n_quantiles: int = 8) -> None:
        super().__init__()
        self.n_cos = n_cos
        self.n_quantiles = n_quantiles
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        # Cosine embedding: τ → cos(i π τ) for i=1..n_cos → linear → feature_dim
        self.cos_embedding = nn.Linear(n_cos, feature_dim)
        # Final Q layer
        self.q_layer = nn.Linear(feature_dim, n_actions)

    def forward(self, features: torch.Tensor, n_quantiles: int | None = None) -> torch.Tensor:
        """Return mean Q-values (B, n_actions) by averaging over sampled quantiles."""
        K = n_quantiles or self.n_quantiles
        B = features.shape[0]

        # Sample τ ~ U(0,1): (B, K)
        if self.training:
            tau = torch.rand(B, K, device=features.device, dtype=features.dtype)
        else:
            # Deterministic quantiles for evaluation
            tau = torch.linspace(0.5 / K, 1.0 - 0.5 / K, K, device=features.device, dtype=features.dtype)
            tau = tau.unsqueeze(0).expand(B, -1)

        # Cosine basis: (B, K, n_cos)
        i_pi = math.pi * torch.arange(1, self.n_cos + 1, device=features.device, dtype=features.dtype)
        cos_features = torch.cos(tau.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0))  # (B, K, n_cos)
        tau_embed = F.relu(self.cos_embedding(cos_features))  # (B, K, feature_dim)

        # Element-wise multiply: (B, 1, feature_dim) * (B, K, feature_dim) → (B, K, feature_dim)
        combined = features.unsqueeze(1) * tau_embed  # (B, K, feature_dim)
        q_quantiles = self.q_layer(combined)  # (B, K, n_actions)

        return q_quantiles.mean(dim=1)  # (B, n_actions)

    def forward_quantiles(self, features: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Given explicit τ (B, K), return quantile Q-values (B, K, n_actions)."""
        i_pi = math.pi * torch.arange(1, self.n_cos + 1, device=features.device, dtype=features.dtype)
        cos_features = torch.cos(tau.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0))
        tau_embed = F.relu(self.cos_embedding(cos_features))
        combined = features.unsqueeze(1) * tau_embed
        return self.q_layer(combined)
