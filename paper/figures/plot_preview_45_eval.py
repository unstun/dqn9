"""预览：4.5 三变体训练评估曲线 (SR + Return + Path Length)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from style import apply_style, RUNS, rolling_mean, C_FULL, C_NOAM, C_NODQFD, FIG_DIR

apply_style()

VARIANTS = {
    "Full (AM+DQfD)": RUNS / "abl_diag10k_kt02_cnn_ddqn_md"
                            / "train_20260325_120844" / "training_eval.csv",
    "w/o AM":         RUNS / "abl_amdqfd_noAM"
                            / "train_20260316_001347" / "training_eval.csv",
    "w/o DQfD":       RUNS / "abl_amdqfd_noDQfD"
                            / "train_20260316_001347" / "training_eval.csv",
}
COLORS = {"Full (AM+DQfD)": C_FULL, "w/o AM": C_NOAM, "w/o DQfD": C_NODQFD}
LW = {"Full (AM+DQfD)": 2.2, "w/o AM": 1.8, "w/o DQfD": 1.8}
WINDOW = 15

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

for label, csv_path in VARIANTS.items():
    df = pd.read_csv(csv_path)
    color, lw = COLORS[label], LW[label]

    # (a) Success Rate
    axes[0].scatter(df["episode"], df["success_rate"], c=color, s=6, alpha=0.2)
    axes[0].plot(df["episode"], rolling_mean(df["success_rate"], WINDOW),
                 color=color, lw=lw, label=label)

    # (b) Average Return
    axes[1].scatter(df["episode"], df["avg_return"], c=color, s=6, alpha=0.2)
    axes[1].plot(df["episode"], rolling_mean(df["avg_return"], WINDOW),
                 color=color, lw=lw, label=label)

    # (c) Path Length (only successful episodes)
    df_ok = df[df["success_rate"] > 0].copy()
    axes[2].scatter(df_ok["episode"], df_ok["avg_path_length"], c=color, s=6, alpha=0.2)
    if len(df_ok) > 3:
        axes[2].plot(df_ok["episode"], rolling_mean(df_ok["avg_path_length"], WINDOW),
                     color=color, lw=lw, label=label)

axes[0].set_title("(a) Success Rate")
axes[0].set_ylabel("Evaluation SR")
axes[0].set_ylim(-0.05, 1.1)

axes[1].set_title("(b) Average Return")
axes[1].set_ylabel("Avg Return")

axes[2].set_title("(c) Path Length (success only)")
axes[2].set_ylabel("Path Length (m)")

for ax in axes:
    ax.set_xlabel("Training episode")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

fig.tight_layout()
out = FIG_DIR / "preview_45_eval.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
