"""4.4 节 TD Loss 训练曲线: 4 种 DQN 架构变体对比。

4 条滚动均值曲线叠加, MD-DQN 加粗置顶。
TD loss 跨数个量级, 采用对数纵轴。

数据源: runs202643/train/abl_arch_cnn_dqn*/training_diagnostics.csv
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from style import apply_style, save_fig, RUNS3, rolling_mean, ARCH_COLORS, ARCH_DIR_TO_LABEL

ROLLING_W = 500

apply_style()

# ── 读取数据 ────────────────────────────────────────────────────────────
curves: dict[str, pd.DataFrame] = {}
for dir_suffix, label in ARCH_DIR_TO_LABEL.items():
    csv_path = RUNS3 / "train" / f"abl_arch_{dir_suffix}" / "training_diagnostics.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    df["td_loss_smooth"] = rolling_mean(df["td_loss"], window=ROLLING_W)
    curves[label] = df

# ── 按最终 loss 排序 (图例用) ───────────────────────────────────────────
sorted_labels = sorted(
    curves.keys(),
    key=lambda lb: curves[lb]["td_loss_smooth"].iloc[-1],
)

# ── 绘图 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))

for label in sorted_labels:
    df = curves[label]
    color = ARCH_COLORS[label]

    # MD-DQN 加粗置顶
    if label == "MD-DQN":
        lw, zorder = 2.5, 10
    else:
        lw, zorder = 1.5, 5

    ax.plot(
        df["episode"],
        df["td_loss_smooth"],
        color=color,
        linewidth=lw,
        linestyle="-",
        zorder=zorder,
        label=label,
    )

# ── 对数纵轴 ─────────────────────────────────────────────────────────
ax.set_yscale("log")

ax.set_xlabel("Training episode")
ax.set_ylabel("TD Loss")
ax.grid(True, alpha=0.3, which="both")

ax.legend(ncol=2, loc="upper right", framealpha=0.9)

fig.tight_layout()
save_fig(fig, "fig_44_td_loss")
plt.close(fig)
