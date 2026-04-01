"""8-variant DRL ablation SR -- horizontal grouped bar chart.

Data: g1t03_s110_all_20260326, seed=110, 50 runs, goal_tolerance=0.3m.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
HIGHLIGHT = "#f0f0f0"

# ── data (sorted by Long SR desc, then Short SR desc) ──
variants  = ["MHA-DDQN", "DQN", "Duel-DDQN", "MHA-DQN",
             "MD-DQN", "Duel-DQN", "MD-DDQN", "DDQN"]
long_sr   = [66, 72, 74, 76, 78, 78, 80, 80]
short_sr  = [66, 68, 74, 70, 84, 80, 72, 60]

n = len(variants)
y = np.arange(n)
bar_h = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

# highlight MD-DDQN row
hi = variants.index("MD-DDQN")
ax.axhspan(hi - 0.45, hi + 0.45, color=HIGHLIGHT, zorder=0)

bars_long  = ax.barh(y + bar_h / 2, long_sr, bar_h,
                     label="Long ($\\geq$18 m)", color=BLUE,
                     edgecolor="white", linewidth=0.5, zorder=2)
bars_short = ax.barh(y - bar_h / 2, short_sr, bar_h,
                     label="Short (6\u201314 m)", color=ORANGE,
                     edgecolor="white", linewidth=0.5, zorder=2)

for bars in (bars_long, bars_short):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{int(w)}%", va="center", ha="left", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(variants)
for i, lbl in enumerate(ax.get_yticklabels()):
    if i == hi:
        lbl.set_fontweight("bold")

ax.set_xlabel("Success Rate (%)")
ax.set_xlim(0, 105)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06),
          ncol=2, frameon=False)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_ablation_sr.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
