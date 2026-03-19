"""带时间戳的实验运行目录管理。

目录约定
--------
runs/<experiment_name>/train_YYYYMMDD_HHMMSS/          <- 单次训练运行
                       train_.../models/<env>/          <- 保存的 checkpoint
                       train_.../infer/YYYYMMDD_HHMMSS/ <- 推理输出
                       latest.txt                       <- 指向最近一次运行

提供
----
- resolve_experiment_dir()      裸名称 -> runs/<name>/；路径 -> 原样使用。
- create_run_dir()              创建带时间戳的新运行目录。
- latest_run_dir()              查找最近的运行目录（通过 latest.txt 或时间戳排序）。
- latest_run_dir_with_models()  查找包含 models/ 的最近运行目录。
- resolve_models_dir()          从实验名称 / 运行目录 / models 路径灵活解析。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_RUN_DIR_RE = re.compile(r"^(?:(?P<prefix>[A-Za-z]+)_)?(?P<ts>\d{8}_\d{6})(?:_(?P<n>\d+))?$")


@dataclass(frozen=True)
class RunPaths:
    experiment_dir: Path
    run_dir: Path


def _run_dir_sort_key(name: str) -> tuple[str, int] | None:
    m = _RUN_DIR_RE.match(name)
    if not m:
        return None
    ts = m.group("ts")
    n = int(m.group("n") or 0)
    return ts, n


def resolve_experiment_dir(out: Path, *, runs_root: Path = Path("runs")) -> Path:
    """解析输出实验目录。

    约定：
    - 若 `out` 是裸名称（如 "outputs_repro_1000"），存储在 `runs/<name>/` 下。
    - 若 `out` 是路径（包含分隔符 / 以 '.' 开头 / 是绝对路径），原样使用。
    """
    out = Path(out)
    if out.is_absolute():
        return out

    out_str = out.as_posix()
    runs_root = Path(runs_root)
    is_bare_name = len(out.parts) == 1 and not out_str.startswith(".") and out.name != runs_root.name
    if is_bare_name:
        return runs_root / out
    return out


def _iter_run_dirs(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        return []
    runs: list[tuple[tuple[str, int], Path]] = []
    for p in experiment_dir.iterdir():
        if not p.is_dir():
            continue
        key = _run_dir_sort_key(p.name)
        if key is None:
            continue
        runs.append((key, p))
    runs.sort(key=lambda pair: pair[0])
    return [p for _, p in runs]


def latest_run_dir(experiment_dir: Path) -> Path | None:
    """返回 `experiment_dir` 下最新的带时间戳运行目录。"""
    latest_file = experiment_dir / "latest.txt"
    if latest_file.exists():
        name = latest_file.read_text(encoding="utf-8").strip()
        if name:
            candidate = experiment_dir / name
            if candidate.is_dir():
                return candidate

    runs = _iter_run_dirs(experiment_dir)
    return runs[-1] if runs else None


def latest_run_dir_with_models(experiment_dir: Path) -> Path | None:
    """返回 `experiment_dir` 下包含 `models/` 的最新运行目录。"""
    candidate = latest_run_dir(experiment_dir)
    if candidate is not None and (candidate / "models").is_dir():
        return candidate

    for run_dir in reversed(_iter_run_dirs(experiment_dir)):
        if (run_dir / "models").is_dir():
            return run_dir
    return None


def create_run_dir(
    experiment_dir: Path,
    *,
    timestamp_runs: bool = True,
    now: datetime | None = None,
    prefix: str | None = None,
) -> RunPaths:
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not timestamp_runs:
        return RunPaths(experiment_dir=experiment_dir, run_dir=experiment_dir)

    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_{ts}" if prefix else ts
    run_dir = experiment_dir / stem
    n = 0
    while run_dir.exists():
        n += 1
        run_dir = experiment_dir / f"{stem}_{n}"

    run_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "latest.txt").write_text(run_dir.name, encoding="utf-8")
    return RunPaths(experiment_dir=experiment_dir, run_dir=run_dir)


def resolve_models_dir(models: Path, *, runs_root: Path = Path("runs")) -> Path:
    """解析用于推理的 models 目录。

    接受以下输入：
    - 实验名称（裸名称）：使用 `runs/<name>/` 下最新运行的 models
    - 实验目录路径：使用 `<dir>/` 下最新运行的 models
    - 运行目录路径：使用 `<run>/models`
    - models 目录路径：直接使用
    """
    raw = Path(models)

    candidates: list[Path] = []
    mapped = resolve_experiment_dir(raw, runs_root=runs_root)
    candidates.append(mapped)

    if raw != mapped:
        candidates.append(raw)
    else:
        # 向后兼容：若用户传入如 "outputs_repro_1000/models"，也尝试在 "runs/..." 下查找
        if not raw.is_absolute() and raw.parts and raw.parts[0] != Path(runs_root).name:
            candidates.append(Path(runs_root) / raw)

    tried: list[str] = []
    for base in candidates:
        tried.append(str(base))

        if base.is_dir():
            if base.name == "models":
                return base
            if (base / "models").is_dir():
                return base / "models"

            lr = latest_run_dir_with_models(base)
            if lr is not None:
                return lr / "models"

        if base.name == "models" and base.parent.is_dir():
            lr = latest_run_dir_with_models(base.parent)
            if lr is not None:
                return lr / "models"

    raise FileNotFoundError(
        "Could not resolve models directory.\n"
        f"- models={raw}\n"
        f"- tried={tried}"
    )
