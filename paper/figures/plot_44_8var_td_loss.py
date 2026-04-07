"""8-Variant Ablation: TD Loss comparison plot (§4.4)."""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "runs" / "abl_8var_diagnostics"

VARIANTS = {
    "cnn_ddqn":           "DQN",
    "cnn_ddqn_duel":      "Duel",
    "cnn_ddqn_noisy":     "Noisy",
    "cnn_ddqn_noisy_duel":"Noisy-Duel",
    "cnn_ddqn_munch":     "M-DQN",
    "cnn_ddqn_munch_duel":"DM-DQN",
}

WINDOW = 100  # 滑动平均窗口

fig, ax = plt.subplots(figsize=(12, 5))

colors = plt.cm.tab10(np.linspace(0, 1, len(VARIANTS)))

for (suffix, label), color in zip(VARIANTS.items(), colors):
    csv_path = DATA_DIR / f"{suffix}.csv"
    if not csv_path.exists():
        print(f"  [skip] {label}: {csv_path} not found")
        continue
    df = pd.read_csv(csv_path)
    td = df["td_loss"].values
    # 滑动平均
    smoothed = pd.Series(td).rolling(WINDOW, min_periods=1).mean().values
    ax.plot(df["episode"].values, smoothed, label=label, color=color, linewidth=1.2)
    # 标注最小 TD loss 点
    min_idx = np.argmin(smoothed)
    min_val = smoothed[min_idx]
    min_ep = df["episode"].values[min_idx]
    ax.plot(min_ep, min_val, "v", color=color, markersize=6)
    print(f"  {label:12s}: min TD = {min_val:.4f} @ ep {int(min_ep)}")

ax.set_xlabel("Episode")
ax.set_ylabel("TD Loss (100-ep moving avg)")
ax.set_title("8-Variant Ablation: TD Loss Comparison")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(0, 10000)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_png = pathlib.Path(__file__).resolve().parent / "fig_44_8var_td_loss.png"
out_pdf = pathlib.Path(__file__).resolve().parent / "fig_44_8var_td_loss.pdf"
fig.savefig(out_png, dpi=200)
fig.savefig(out_pdf)
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
