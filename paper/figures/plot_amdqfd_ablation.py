"""AM x DQfD component ablation figure.

Data: g1t03_s110_all_20260326, seed=110, 50 runs.
Quality: 3-variant all-succeed subset (N=6 Long, N=15 Short).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── unified style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#2166ac"
ORANGE = "#ef8a62"
RED = "#b2182b"
colors = [BLUE, ORANGE, RED]

variants = ["Full\n(AM+DQfD)", "w/o AM", "w/o DQfD"]

# ── data ──
sr_long  = [80, 48, 28]
sr_short = [72, 62, 40]

# Long quality (N=6)
pathlen = [24.660, 24.949, 25.579]
curv    = [0.1231, 0.1576, 0.1488]

# ── figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ===== Subplot 1: Success Rate =====
x_groups = np.arange(2)
bar_w = 0.22
offsets = [-bar_w, 0, bar_w]

for i, (var, c) in enumerate(zip(variants, colors)):
    vals = [sr_long[i], sr_short[i]]
    bars = ax1.bar(x_groups + offsets[i], vals, bar_w,
                   label=var, color=c, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                 f"{v}%", ha="center", va="bottom", fontsize=9)

# delta annotations (Long)
ax1.annotate("\u221232 pp", xy=(0 + offsets[1], sr_long[1]),
             xytext=(0 + offsets[1], sr_long[1] + 14),
             ha="center", va="bottom", fontsize=8, color=ORANGE,
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
ax1.annotate("\u221252 pp", xy=(0 + offsets[2], sr_long[2]),
             xytext=(0 + offsets[2] + 0.06, sr_long[2] + 22),
             ha="center", va="bottom", fontsize=8, color=RED,
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax1.set_xticks(x_groups)
ax1.set_xticklabels(["Long ($\\geq$18 m)", "Short (6\u201314 m)"])
ax1.set_ylabel("Success Rate (%)")
ax1.set_ylim(0, 115)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax1.set_title("(a) Success Rate", fontsize=12, fontweight="bold", pad=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ===== Subplot 2: Long-range Path Quality =====
x = np.arange(len(variants))
bar_w2 = 0.55

bars2 = ax2.bar(x, pathlen, bar_w2, color=colors,
                edgecolor="white", linewidth=0.5)
ax2.set_ylabel("Path Length (m)")
ax2.set_ylim(24.0, 26.5)
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

for bar, pl in zip(bars2, pathlen):
    ax2.text(bar.get_x() + bar.get_width() / 2, pl + 0.03,
             f"{pl:.3f} m", ha="center", va="bottom", fontsize=8.5)

ax2.annotate("+0.289 m", xy=(1, pathlen[1]),
             xytext=(1.28, pathlen[1] + 0.65),
             ha="center", va="bottom", fontsize=8, color=ORANGE,
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
ax2.annotate("+0.919 m", xy=(2, pathlen[2]),
             xytext=(2.32, pathlen[2] + 0.62),
             ha="center", va="bottom", fontsize=8, color=RED,
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# secondary axis: curvature
ax2r = ax2.twinx()
ax2r.plot(x, curv, "ko-", markersize=7, linewidth=1.8, zorder=5,
          label="Curvature")
for xi, cv in enumerate(curv):
    ax2r.text(xi + 0.12, cv - 0.008, f"{cv:.4f}", ha="left", va="top",
              fontsize=8, fontstyle="italic")
ax2r.set_ylabel("Mean Curvature (1/m)")
ax2r.set_ylim(0.06, 0.22)
ax2r.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax2r.spines["top"].set_visible(False)

ax2.set_xticks(x)
ax2.set_xticklabels(variants)
ax2.set_title("(b) Long-range Path Quality ($N$=6)",
              fontsize=12, fontweight="bold", pad=10)
ax2.spines["top"].set_visible(False)
ax2r.legend(loc="upper left", fontsize=9, framealpha=0.9)

fig.tight_layout(w_pad=3.0)
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig_amdqfd_ablation.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
