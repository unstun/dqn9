"""4.5 AM x DQfD 组件消融 -- TD Loss 训练曲线: 3 变体对比。

Full (AM+DQfD) / w/o AM / w/o DQfD 叠加显示，
散点 (alpha=0.08) 背景 + 滚动均值实线。

数据源: runs202643/train/abl_arch_cnn_dqn_md/training_diagnostics.csv  (Full)
         runs202643/train/abl_amdqfd_dqn_noAM/training_diagnostics.csv
         runs202643/train/abl_amdqfd_dqn_noDQfD/training_diagnostics.csv
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from style import apply_style, save_fig, RUNS3, rolling_mean, C_FULL, C_NOAM, C_NODQFD

apply_style()

# ── 数据路径 ────────────────────────────────────────────────────────────
CSV_PATHS = {
    "Full (AM+DQfD)": RUNS3 / "train" / "abl_arch_cnn_dqn_md"
                             / "training_diagnostics.csv",
    "w/o AM":         RUNS3 / "train" / "abl_amdqfd_dqn_noAM"
                             / "training_diagnostics.csv",
    "w/o DQfD":       RUNS3 / "train" / "abl_amdqfd_dqn_noDQfD"
                             / "training_diagnostics.csv",
}

COLORS = {"Full (AM+DQfD)": C_FULL, "w/o AM": C_NOAM, "w/o DQfD": C_NODQFD}
LW     = {"Full (AM+DQfD)": 2.2,    "w/o AM": 1.8,    "w/o DQfD": 1.8}

ROLLING_W = 500


# ── 主流程 ──────────────────────────────────────────────────────────────
def main():
    curves: dict[str, pd.DataFrame] = {}
    for label, path in CSV_PATHS.items():
        df = pd.read_csv(path)
        if df["td_loss"].isna().all() or (df["td_loss"] == 0).all():
            df["td_loss"] = df["loss"]
        df["td_smooth"] = rolling_mean(df["td_loss"].dropna(), window=ROLLING_W)
        curves[label] = df

    # ── 判断是否需要 log 纵轴 ──
    all_max = max(c["td_loss"].dropna().max() for c in curves.values())
    all_min = min(c["td_loss"].dropna().clip(lower=1e-6).min()
                  for c in curves.values())
    use_log = (all_max / max(all_min, 1e-12)) > 100

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(7, 3.0))

    draw_order = ["w/o DQfD", "w/o AM", "Full (AM+DQfD)"]
    for label in draw_order:
        df = curves[label]
        color = COLORS[label]

        ax.scatter(
            df["episode"], df["td_loss"],
            c=color, alpha=0.08, s=4, edgecolors="none",
            rasterized=True, zorder=2,
        )

        td_smooth = rolling_mean(df["td_loss"], window=ROLLING_W)
        ax.plot(
            df["episode"], td_smooth,
            color=color, linewidth=LW[label],
            label=label, zorder=5 if label.startswith("Full") else 4,
        )

    if use_log:
        ax.set_yscale("log")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("TD Loss")
    ax.grid(True, alpha=0.3, which="both" if use_log else "major")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_45_td_loss")
    plt.close(fig)


if __name__ == "__main__":
    main()
