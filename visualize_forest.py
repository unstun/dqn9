from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize forest_* occupancy grids (focus: forest_a).")
    ap.add_argument("--env", type=str, default="forest_a", help="Environment name (e.g., forest_a).")
    ap.add_argument("--out", type=Path, default=None, help="Optional output image path (.png).")
    ap.add_argument("--dpi", type=int, default=200, help="Saved image DPI (when --out is set).")
    ap.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show an interactive matplotlib window (disable with --no-show).",
    )
    ap.add_argument(
        "--grid-lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw light grid lines (slow for large maps).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ugv_dqn.maps import get_map_spec  # local import after sys.path patch

    env_name = str(args.env).strip()
    spec = get_map_spec(env_name)
    grid = spec.obstacle_grid().astype(np.uint8, copy=False)  # (H,W) y=0 bottom, 1=obstacle
    h, w = grid.shape

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))

    ax.imshow(grid, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.scatter([float(spec.start_xy[0])], [float(spec.start_xy[1])], marker="*", s=140, color="dodgerblue", label="Start")
    ax.scatter([float(spec.goal_xy[0])], [float(spec.goal_xy[1])], marker="*", s=140, color="crimson", label="Goal")
    ax.set_title(f"{env_name} (H={h}, W={w})")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")

    if bool(args.grid_lines):
        ax.set_xticks(np.arange(-0.5, w, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1.0), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.2, alpha=0.08)
        ax.tick_params(which="minor", bottom=False, left=False)

    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(args.dpi))
        print(f"Wrote: {out_path}")

    if bool(args.show):
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
