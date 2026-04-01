"""Heatmap of 8 DRL variants x 6 metrics (ablation study).

Data: g1t03_s110_all_20260326, seed=110, 50 runs.
SR from mean_raw; Quality from 8-variant all-succeed subset (N=16 Long, N=14 Short).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── unified style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── data (initial order matches quality table, will be re-sorted) ──
variants = [
    "MD-DDQN", "MHA-DQN", "Duel-DDQN", "MHA-DDQN",
    "MD-DQN", "Duel-DQN", "DQN", "DDQN",
]
metrics = ["Long SR", "Short SR", "Long PL", "Short PL",
           "Long Curv", "Short Curv"]

# cols: Long SR(%), Short SR(%), Long PL(m), Short PL(m), Long Curv, Short Curv
raw = np.array([
    [80, 72, 26.819, 9.294, 0.1407, 0.1394],  # MD-DDQN
    [76, 70, 26.828, 9.327, 0.1379, 0.1554],  # MHA-DQN
    [74, 74, 26.831, 9.319, 0.1452, 0.1438],  # Duel-DDQN
    [66, 66, 26.872, 9.354, 0.1489, 0.1689],  # MHA-DDQN
    [78, 84, 26.876, 9.405, 0.1395, 0.1626],  # MD-DQN
    [78, 80, 26.930, 9.309, 0.1487, 0.1488],  # Duel-DQN
    [72, 68, 26.943, 9.286, 0.1510, 0.1790],  # DQN
    [80, 60, 26.982, 9.359, 0.1553, 0.1533],  # DDQN
])

# ── normalize each column to [0,1], 1 = best ──
norm = np.zeros_like(raw)
for j in range(raw.shape[1]):
    col = raw[:, j]
    if j < 2:  # SR: higher = better
        norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-12)
    else:  # PL, Curv: lower = better
        norm[:, j] = (col.max() - col) / (col.max() - col.min() + 1e-12)

# ── sort by average normalized score descending ──
avg = norm.mean(axis=1)
order = np.argsort(-avg)
raw = raw[order]
norm = norm[order]
variants = [variants[i] for i in order]

# ── plot ──
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(metrics, fontsize=12)
ax.set_yticks(np.arange(len(variants)))
ax.set_yticklabels(variants, fontsize=12)

for i in range(len(variants)):
    for j in range(len(metrics)):
        val = raw[i, j]
        if j < 2:
            txt = f"{int(val)}%"
        elif j < 4:
            txt = f"{val:.2f}"
        else:
            txt = f"{val:.4f}"
        tc = "white" if norm[i, j] < 0.3 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=tc)

# highlight MD-DDQN
md_row = variants.index("MD-DDQN")
rect = Rectangle((-0.5, md_row - 0.5), len(metrics), 1,
                 linewidth=3, edgecolor="#2166ac", facecolor="none", zorder=5)
ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Normalized Score (1 = best)", fontsize=11)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_ablation_heatmap.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
