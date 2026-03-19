"""MD-DDQN (MHA + Dueling + Double DQN) CNN 架构图 — 自上而下布局。

布局：自上而下流式
  两个输入 → CNN → MHA → Concat → Shared FC → V/A 分叉 → Q
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "text.usetex": False,
})

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(-1, 13)
ax.set_ylim(-1, 15)
ax.set_aspect("equal")
ax.axis("off")

# ── 颜色 ──
C_INPUT  = "#DBEAFE"
C_CONV   = "#D1FAE5"
C_MHA    = "#FEF3C7"
C_FC     = "#EDE9FE"
C_OUTPUT = "#FEE2E2"
C_CONCAT = "#E5E7EB"
C_EDGE   = "#374151"
C_TEXT   = "#111827"


def box(cx, cy, w, h, color, lines, fontsize=9, bold_idx=0):
    """以 (cx,cy) 为中心绘制圆角矩形。"""
    x, y = cx - w / 2, cy - h / 2
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=C_EDGE, linewidth=1.3,
    )
    ax.add_patch(patch)
    n = len(lines)
    gap = 0.32
    y_start = cy + (n - 1) * gap / 2
    for i, line in enumerate(lines):
        weight = "bold" if i == bold_idx else "normal"
        fs = fontsize if i == bold_idx else fontsize - 1.5
        c = C_TEXT if i == bold_idx else "#6B7280"
        ax.text(cx, y_start - i * gap, line,
                ha="center", va="center", fontsize=fs,
                fontweight=weight, color=c)
    return cx, cy, w, h


def varrow(x1, y1, x2, y2, label=None, rad=0, label_side="right"):
    """垂直方向箭头。"""
    conn = f"arc3,rad={rad}"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.4,
                                connectionstyle=conn))
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset = 0.15 if label_side == "right" else -0.15
        ha = "left" if label_side == "right" else "right"
        ax.text(mx + offset, my, label, fontsize=7, color="#9CA3AF",
                ha=ha, va="center", style="italic")


# ── 中心 x 坐标 ──
CX = 6.0       # 主干中心
CX_L = 3.5     # 标量分支
CX_V = 4.2     # Value stream
CX_A = 7.8     # Advantage stream

BW = 3.2       # 主干盒宽
BH = 0.85      # 盒高

# ============================================================
#  第 1 层: 输入
# ============================================================
y_in = 13.5

# 地图通道输入
box(CX + 1.2, y_in, 3.0, 1.0, C_INPUT,
    ["Map Channels", "3 × 12 × 12", "occupancy / goal distance / EDT"])

# 标量输入
box(CX - 2.8, y_in, 2.5, 1.0, C_INPUT,
    ["Scalar Features", "11-dim", "pose, v, δ, d_goal …"])

# ============================================================
#  第 2 层: CNN Backbone
# ============================================================
y_c1 = 11.8
box(CX + 1.2, y_c1, BW, BH, C_CONV,
    ["Conv2d  3 → 32", "3×3, stride=1, pad=1, ReLU"])
varrow(CX + 1.2, y_in - 0.5, CX + 1.2, y_c1 + BH / 2, label="3×12×12")

y_c2 = 10.5
box(CX + 1.2, y_c2, BW, BH, C_CONV,
    ["Conv2d  32 → 64", "3×3, stride=2, pad=1, ReLU"])
varrow(CX + 1.2, y_c1 - BH / 2, CX + 1.2, y_c2 + BH / 2, label="32×12×12")

y_c3 = 9.2
box(CX + 1.2, y_c3, BW, BH, C_CONV,
    ["Conv2d  64 → 64", "3×3, stride=2, pad=1, ReLU"])
varrow(CX + 1.2, y_c2 - BH / 2, CX + 1.2, y_c3 + BH / 2, label="64×6×6")

# ============================================================
#  第 3 层: SpatialMHA
# ============================================================
y_mha = 7.7
box(CX + 1.2, y_mha, BW + 0.3, 1.0, C_MHA,
    ["SpatialMHA", "9 tokens × 64-dim, 4 heads", "Self-Attention + Residual + LayerNorm"])
varrow(CX + 1.2, y_c3 - BH / 2, CX + 1.2, y_mha + 0.5, label="64×3×3")

# ============================================================
#  Flatten + Concat
# ============================================================
y_fl = 6.3
box(CX + 1.2, y_fl, 2.2, 0.7, C_CONCAT,
    ["Flatten → 576"])
varrow(CX + 1.2, y_mha - 0.5, CX + 1.2, y_fl + 0.35)

y_cat = 5.2
box(CX, y_cat, 2.8, 0.7, C_CONCAT,
    ["Concat → 587-dim", "11 (scalar) + 576 (conv)"], fontsize=8)
# Flatten → Concat
varrow(CX + 1.2, y_fl - 0.35, CX + 0.3, y_cat + 0.35, rad=0.1)
# Scalar → Concat（从左上到中间，弯曲箭头）
varrow(CX - 2.8, y_in - 0.5, CX - 0.3, y_cat + 0.35, rad=-0.15)

# ============================================================
#  Dueling Head
# ============================================================

# Shared FC
y_sh = 3.8
box(CX, y_sh, BW, BH, C_FC,
    ["Shared FC", "587 → 256, ReLU"])
varrow(CX, y_cat - 0.35, CX, y_sh + BH / 2)

# Value stream
y_va = 2.3
box(CX_V, y_va, 2.2, BH, C_FC,
    ["Value Stream", "256 → 256 → 1"])
varrow(CX - 0.5, y_sh - BH / 2, CX_V, y_va + BH / 2, rad=0.1)

# Advantage stream
box(CX_A, y_va, 2.5, BH, C_FC,
    ["Advantage Stream", "256 → 256 → 35"])
varrow(CX + 0.5, y_sh - BH / 2, CX_A, y_va + BH / 2, rad=-0.1)

# Q output
y_q = 0.8
box(CX, y_q, 4.5, 0.85, C_OUTPUT,
    ["Q(s,a) = V(s) + A(s,a) − mean(A)", "→ 35 discrete actions"],
    fontsize=9.5)
varrow(CX_V, y_va - BH / 2, CX - 0.5, y_q + 0.85 / 2, rad=0.1)
varrow(CX_A, y_va - BH / 2, CX + 0.5, y_q + 0.85 / 2, rad=-0.1)

# ============================================================
#  区域标注（右侧竖排文字）
# ============================================================
def side_label(y_top, y_bot, label, color):
    ym = (y_top + y_bot) / 2
    ax.text(10.5, ym, label, fontsize=10, fontweight="bold",
            color=color, ha="center", va="center", rotation=-90, alpha=0.7)
    # 竖线
    ax.plot([10.1, 10.1], [y_bot, y_top], color=color, lw=2, alpha=0.3)
    ax.plot([10.1, 10.3], [y_top, y_top], color=color, lw=2, alpha=0.3)
    ax.plot([10.1, 10.3], [y_bot, y_bot], color=color, lw=2, alpha=0.3)

side_label(y_c1 + BH / 2 + 0.1, y_c3 - BH / 2 - 0.1, "CNN", "#059669")
side_label(y_mha + 0.5 + 0.1, y_mha - 0.5 - 0.1, "MHA", "#D97706")
side_label(y_sh + BH / 2 + 0.1, y_va - BH / 2 - 0.1, "DUELING", "#7C3AED")

# ============================================================
#  标题
# ============================================================
ax.text(CX, -0.5, "MD-DDQN Network Architecture",
        fontsize=14, fontweight="bold", ha="center", color=C_TEXT)
ax.text(CX, -0.95, "Multi-Head Attention + Dueling + Double DQN",
        fontsize=10, ha="center", color="#6B7280")

# ============================================================
#  图例
# ============================================================
legend_patches = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor=C_EDGE, label="Input"),
    mpatches.Patch(facecolor=C_CONV, edgecolor=C_EDGE, label="Conv + ReLU"),
    mpatches.Patch(facecolor=C_MHA, edgecolor=C_EDGE, label="Spatial Multi-Head Attention"),
    mpatches.Patch(facecolor=C_CONCAT, edgecolor=C_EDGE, label="Flatten / Concat"),
    mpatches.Patch(facecolor=C_FC, edgecolor=C_EDGE, label="Dueling Head (FC)"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor=C_EDGE, label="Q-value Output"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
          framealpha=0.95, edgecolor="#D1D5DB",
          bbox_to_anchor=(-0.05, 1.02))

plt.tight_layout()
plt.savefig("paper/media/fig_md_ddqn_arch.pdf", dpi=300, bbox_inches="tight")
plt.savefig("paper/media/fig_md_ddqn_arch.png", dpi=200, bbox_inches="tight")
print("Saved: paper/media/fig_md_ddqn_arch.{pdf,png}")
