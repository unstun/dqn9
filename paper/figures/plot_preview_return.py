"""快速预览：8 个架构变体的 episode return 训练曲线。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from style import apply_style, RUNS, ARCH_DIR_TO_LABEL, ARCH_COLORS, rolling_mean, FIG_DIR

apply_style()

TRAIN_TS = "train_20260325_120844"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

for dir_suffix, label in ARCH_DIR_TO_LABEL.items():
    csv = RUNS / f"abl_diag10k_kt02_{dir_suffix}" / TRAIN_TS / "training_eval.csv"
    df = pd.read_csv(csv)
    color = ARCH_COLORS[label]
    lw = 2.5 if label == "MD-DDQN" else 1.3
    ls = "-" if "DDQN" in label else "--"
    zorder = 10 if label == "MD-DDQN" else 5

    # (a) avg_return
    smooth = rolling_mean(df["avg_return"], window=25)
    ax1.plot(df["episode"], smooth, color=color, lw=lw, ls=ls, label=label, zorder=zorder)

    # (b) success_rate
    smooth_sr = rolling_mean(df["success_rate"], window=25)
    ax2.plot(df["episode"], smooth_sr, color=color, lw=lw, ls=ls, label=label, zorder=zorder)

ax1.set_xlabel("Training episode")
ax1.set_ylabel("Average Return")
ax1.set_title("(a) Episode Return")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=7, ncol=2, loc="lower right")

ax2.set_xlabel("Training episode")
ax2.set_ylabel("Success Rate")
ax2.set_title("(b) Evaluation Success Rate")
ax2.set_ylim(-0.05, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7, ncol=2, loc="lower right")

fig.tight_layout()
out = FIG_DIR / "preview_8arch_return.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
