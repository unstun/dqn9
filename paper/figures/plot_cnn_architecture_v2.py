#!/usr/bin/env python3
"""
CNN-MD-DDQN 架构可视化（v2，与代码严格一致）

架构路径（Dueling + MHA，即 MD-DDQN）：
  地图 3×12×12 → Conv(3→32) → Conv(32→64,s2) → Conv(64→64,s2)
               → SpatialMHA(h=4) → Flatten(576)
  标量 11-d ──────────────────────────────────→ (直接拼接，无FC)
  拼接 587-d → SharedFC(587→256) → ReLU
            ├→ ValueFC(256→256→1)
            └→ AdvFC(256→256→35)
  Q = V + A - mean(A) → ActionMask → a*

输出: paper/figures/cnn_md_ddqn_architecture.{pdf,png}
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle
import numpy as np
from pathlib import Path


# ── 颜色方案 ──────────────────────────────────────────────────────────
C = {
    "conv": "#5DADE2",
    "conv_b": "#2471A3",
    "mha": "#17A2B8",
    "mha_b": "#117A8B",
    "fc": "#58D68D",
    "fc_b": "#1E8449",
    "value": "#E74C3C",
    "value_b": "#922B21",
    "adv": "#27AE60",
    "adv_b": "#1E8449",
    "out": "#9B59B6",
    "out_b": "#6C3483",
    "mask": "#F39C12",
    "mask_b": "#D68910",
    "concat": "#F4D03F",
    "concat_b": "#B7950B",
    "input": "#E8E8E8",
    "arrow": "#333333",
    "scalar_bg": "#FEF9E7",
    "scalar_b": "#B7950B",
}


def lighten(color, amount=0.3):
    import matplotlib.colors as mc
    import colorsys
    r, g, b = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1.0, l + amount * (1 - l)), s)


def darken(color, amount=0.2):
    import matplotlib.colors as mc
    import colorsys
    r, g, b = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l * (1 - amount), s)


def is_dark(color):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


# ── 绘图原语 ──────────────────────────────────────────────────────────
def draw_3d_block(ax, x, y, w, h, d, label, fc, bc, fontsize=9, sublabel=None, sublabel2=None):
    """绘制3D立方体块"""
    dx, dy = d * 0.5, d * 0.5
    tc, sc = lighten(fc, 0.3), darken(fc, 0.2)

    front = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                    facecolor=fc, edgecolor=bc, lw=1.5, alpha=0.9)
    top = Polygon([(x, y+h), (x+dx, y+h+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                  facecolor=tc, edgecolor=bc, lw=1, alpha=0.9)
    side = Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                   facecolor=sc, edgecolor=bc, lw=1, alpha=0.9)
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)

    txt_color = "white" if is_dark(fc) else "black"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=txt_color)
    if sublabel:
        ax.text(x + w/2, y - 0.18, sublabel, ha="center", fontsize=6.5, color=bc)
    if sublabel2:
        ax.text(x + w/2, y - 0.38, sublabel2, ha="center", fontsize=6, color="#666")
    return x + w/2, y + h/2


def draw_flat(ax, x, y, w, h, label, fc, bc, fontsize=8):
    """绘制扁平圆角块"""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=fc, edgecolor=bc, lw=1.5, alpha=0.9)
    ax.add_patch(rect)
    txt_color = "white" if is_dark(fc) else "black"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=txt_color)
    return x + w/2, y + h/2


def draw_arrow(ax, start, end, color=None, lw=1.8):
    color = color or C["arrow"]
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=14))


def draw_concat(ax, x, y, r=0.22):
    circle = Circle((x, y), r, facecolor="white", edgecolor=C["concat_b"], lw=2)
    ax.add_patch(circle)
    ax.text(x, y, "⊕", ha="center", va="center", fontsize=13, fontweight="bold", color="#333")
    return x, y


def draw_input_maps(ax, x, y):
    """3通道12×12地图输入"""
    sz = 0.9
    gap = 0.12
    labels = ["Occ", "GDF", "EDT"]
    colors = ["#D5D8DC", "#F5B041", "#85C1E9"]
    for i, (lb, co) in enumerate(zip(labels, colors)):
        cx = x + i * (sz + gap)
        rect = FancyBboxPatch((cx, y), sz, sz, boxstyle="round,pad=0.01",
                              facecolor=co, edgecolor="#333", lw=1, alpha=0.8)
        ax.add_patch(rect)
        # 占据栅格用小方块暗示
        if lb == "Occ":
            for j in range(4):
                for k in range(4):
                    rx = cx + 0.08 + j * 0.19
                    ry = y + 0.08 + k * 0.19
                    dot = Circle((rx + 0.04, ry + 0.04), 0.025,
                                 facecolor="#1C2833", alpha=0.6)
                    ax.add_patch(dot)
        ax.text(cx + sz/2, y - 0.13, lb, ha="center", fontsize=6.5, color="#555")
    ax.text(x + 1.5*(sz + gap) - gap/2, y - 0.38, "3 × 12 × 12",
            ha="center", fontsize=7.5, fontweight="bold", color="#333")
    center_x = x + 1.5*(sz + gap) - gap/2
    return center_x, y + sz/2


def draw_scalar_input(ax, x, y):
    """11维标量向量"""
    n = 11
    cw = 0.25
    h = 0.4
    for i in range(n):
        co = "#F9E79F" if i < 2 else ("#ABEBC6" if i < 5 else "#D6EAF8")
        rect = Rectangle((x + i*cw, y), cw - 0.02, h,
                          facecolor=co, edgecolor="#333", lw=0.5)
        ax.add_patch(rect)
    total_w = n * cw
    ax.text(x + total_w/2, y - 0.18, "11-d scalar", ha="center",
            fontsize=7.5, fontweight="bold", color="#333")
    return x + total_w/2, y + h/2


# ── 主绘图 ────────────────────────────────────────────────────────────
def draw_architecture():
    fig, ax = plt.subplots(figsize=(18, 7.5), dpi=300)
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(-0.3, 7.5)
    ax.axis("off")
    ax.set_aspect("equal")

    # 标题
    ax.text(9, 7.1, "CNN-MD-DDQN Architecture",
            ha="center", fontsize=15, fontweight="bold", color="#222")

    # ── 行1：地图分支（y ≈ 4.5） ──
    row_map = 4.3

    # 输入地图
    map_cx, map_cy = draw_input_maps(ax, 0.0, row_map)

    # Conv1
    c1x = 3.5
    c1 = draw_3d_block(ax, c1x, row_map - 0.1, 1.2, 1.2, 0.4, "32",
                       C["conv"], C["conv_b"], 11,
                       sublabel="Conv 3×3, s=1", sublabel2="12 × 12")

    # Conv2
    c2x = c1x + 1.9
    c2 = draw_3d_block(ax, c2x, row_map + 0.05, 1.0, 1.0, 0.4, "64",
                       C["conv"], C["conv_b"], 11,
                       sublabel="Conv 3×3, s=2", sublabel2="6 × 6")

    # Conv3
    c3x = c2x + 1.7
    c3 = draw_3d_block(ax, c3x, row_map + 0.15, 0.8, 0.8, 0.4, "64",
                       C["conv"], C["conv_b"], 10,
                       sublabel="Conv 3×3, s=2", sublabel2="3 × 3")

    # MHA
    mha_x = c3x + 1.5
    mha = draw_flat(ax, mha_x, row_map + 0.25, 0.8, 0.6, "MHA\nh=4",
                    C["mha"], C["mha_b"], 7)

    # Flatten
    fl_x = mha_x + 1.2
    fl = draw_flat(ax, fl_x, row_map + 0.25, 0.9, 0.55, "Flatten\n576",
                   lighten(C["conv"], 0.15), C["conv_b"], 7)

    # 箭头：地图分支
    draw_arrow(ax, (map_cx + 0.5, map_cy), (c1x, c1[1]))
    draw_arrow(ax, (c1x + 1.2, c1[1]), (c2x, c2[1]))
    draw_arrow(ax, (c2x + 1.0, c2[1]), (c3x, c3[1]))
    draw_arrow(ax, (c3x + 0.8, c3[1]), (mha_x, mha[1]))
    draw_arrow(ax, (mha_x + 0.8, mha[1]), (fl_x, fl[1]))

    # ── 行2：标量分支（y ≈ 2.2） ──
    row_scalar = 2.2
    sc_cx, sc_cy = draw_scalar_input(ax, 0.3, row_scalar)

    # 虚线标注"直接拼接，无FC"
    ax.annotate("no FC", xy=(sc_cx + 1.5, sc_cy), fontsize=7,
                fontstyle="italic", color="#999", ha="center")

    # ── 拼接节点 ──
    concat_x = fl_x + 1.5
    concat_y = 3.5
    concat = draw_concat(ax, concat_x, concat_y)

    # 箭头到拼接
    draw_arrow(ax, (fl_x + 0.9, fl[1]), (concat_x - 0.22, concat_y + 0.12))
    draw_arrow(ax, (sc_cx + 1.4, sc_cy), (concat_x - 0.22, concat_y - 0.12))

    # 拼接维度标注
    ax.text(concat_x, concat_y - 0.4, "587-d", ha="center",
            fontsize=7, fontweight="bold", color="#666")

    # ── Shared FC ──
    sfc_x = concat_x + 0.6
    sfc_y = 3.15
    sfc = draw_flat(ax, sfc_x, sfc_y, 1.1, 0.7, "FC\n587→256",
                    C["fc"], C["fc_b"], 7.5)
    draw_arrow(ax, (concat_x + 0.22, concat_y), (sfc_x, sfc[1]))

    # ── 分叉点 ──
    fork_x = sfc_x + 1.1
    fork_y = sfc[1]

    # ── Value 流（上方） ──
    v1_x = fork_x + 0.3
    v1_y = fork_y + 0.9
    v1 = draw_flat(ax, v1_x, v1_y, 1.0, 0.55, "FC\n256→256",
                   C["value"], C["value_b"], 7)
    v2_x = v1_x + 1.3
    v2 = draw_flat(ax, v2_x, v1_y, 0.9, 0.55, "FC\n256→1",
                   C["value"], C["value_b"], 7)
    # V(s) 标签
    ax.text(v1_x + 1.1, v1_y + 0.75, "Value stream  V(s)",
            fontsize=7.5, fontweight="bold", color=C["value_b"])

    draw_arrow(ax, (fork_x, fork_y), (v1_x, v1[1]))
    draw_arrow(ax, (v1_x + 1.0, v1[1]), (v2_x, v2[1]))

    # ── Advantage 流（下方） ──
    a1_x = fork_x + 0.3
    a1_y = fork_y - 1.4
    a1 = draw_flat(ax, a1_x, a1_y, 1.0, 0.55, "FC\n256→256",
                   C["adv"], C["adv_b"], 7)
    a2_x = a1_x + 1.3
    a2 = draw_flat(ax, a2_x, a1_y, 0.9, 0.55, "FC\n256→35",
                   C["adv"], C["adv_b"], 7)
    # A(s,a) 标签
    ax.text(a1_x + 1.1, a1_y - 0.3, "Advantage stream  A(s, a)",
            fontsize=7.5, fontweight="bold", color=C["adv_b"])

    draw_arrow(ax, (fork_x, fork_y), (a1_x, a1[1]))
    draw_arrow(ax, (a1_x + 1.0, a1[1]), (a2_x, a2[1]))

    # ── Q 聚合 ──
    q_x = a2_x + 1.3
    q_y = fork_y - 0.35
    q = draw_flat(ax, q_x, q_y, 1.5, 0.7, "Q = V + A − Ā\n35-d",
                  C["out"], C["out_b"], 7.5)

    draw_arrow(ax, (v2_x + 0.9, v2[1]), (q_x, q[1] + 0.15))
    draw_arrow(ax, (a2_x + 0.9, a2[1]), (q_x, q[1] - 0.15))

    # ── Action Mask ──
    am_x = q_x + 1.8
    am_y = q_y - 0.05
    am = draw_flat(ax, am_x, am_y, 1.2, 0.7, "Action\nMask",
                   C["mask"], C["mask_b"], 8)
    draw_arrow(ax, (q_x + 1.5, q[1]), (am_x, am[1]))

    # ── 输出 ──
    out_x = am_x + 1.5
    out_y = am_y + 0.05
    out = draw_flat(ax, out_x, out_y, 0.8, 0.6, "a*",
                    C["out"], C["out_b"], 10)
    draw_arrow(ax, (am_x + 1.2, am[1]), (out_x, out[1]))

    # ── 每层 ReLU 标注（简洁标注在箭头上方） ──
    relu_positions = [
        ((c1x + 1.2 + c2x) / 2, c1[1] + 0.25),
        ((c2x + 1.0 + c3x) / 2, c2[1] + 0.2),
        ((c3x + 0.8 + mha_x) / 2, c3[1] + 0.2),
        (sfc_x + 0.55, sfc_y + 0.85),
        ((v1_x + 1.0 + v2_x) / 2, v1_y + 0.65),
        ((a1_x + 1.0 + a2_x) / 2, a1_y + 0.65),
    ]
    for rx, ry in relu_positions:
        ax.text(rx, ry, "ReLU", fontsize=5.5, ha="center", color="#888",
                fontstyle="italic")

    # ── 图例 ──
    legend_y = -0.05
    legend_items = [
        ("Conv Layer", C["conv"]),
        ("Spatial MHA", C["mha"]),
        ("Shared FC", C["fc"]),
        ("Value V(s)", C["value"]),
        ("Advantage A(s,a)", C["adv"]),
        ("Output / Q", C["out"]),
        ("Action Mask", C["mask"]),
    ]
    for i, (lb, co) in enumerate(legend_items):
        lx = 1.5 + i * 2.2
        rect = Rectangle((lx, legend_y), 0.3, 0.2,
                          facecolor=co, edgecolor="#333", lw=0.5)
        ax.add_patch(rect)
        ax.text(lx + 0.4, legend_y + 0.1, lb, va="center", fontsize=6.5)

    plt.tight_layout()
    return fig


def main():
    fig = draw_architecture()
    out_dir = Path(__file__).parent
    stem = "cnn_md_ddqn_architecture"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", format="pdf")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300, format="png")
    print(f"Saved: {out_dir / stem}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
