"""JSON 配置文件加载与 argparse 集成。

提供
----
- load_json()              读取 JSON 文件并返回字典。
- resolve_config_path()    将 --config / --profile 解析为 configs/ 下的 Path。
- select_section()         从配置中提取 "train" 或 "infer" 子字典。
- apply_config_defaults()  将配置字典的值合并到 argparse 解析器的默认值中，
                           并根据每个参数声明的类型进行类型转换。
- parser_defaults()        将所有 argparse 默认值导出为 JSON 兼容的字典。

配置解析优先级
--------------
1. --profile <name>  → configs/<name>.json
2. --config <path>   → 直接使用路径（回退到 configs/ 前缀查找）
3. （都未指定）      → 若 configs/config.json 存在则使用，否则返回 None
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_compatible(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_compatible(v) for v in value]
    if isinstance(value, list):
        return [_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    return value


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object at top-level: {path}")
    return raw


def resolve_config_path(
    *,
    config: Path | None,
    profile: str | None,
    default_path: Path = Path("configs/config.json"),
    profiles_dir: Path = Path("configs"),
) -> Path | None:
    if config is not None and profile is not None:
        raise ValueError("Use only one of --config or --profile.")

    if profile is not None:
        p = Path(str(profile).strip())
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
            if p.suffix == "":
                candidates.append(p.with_suffix(".json"))
        else:
            candidates.append(profiles_dir / p)
            if p.suffix == "":
                candidates.append((profiles_dir / p).with_suffix(".json"))

        for cand in candidates:
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Config profile not found: {profile!r} (looked for {', '.join(map(str, candidates))})")

    if config is not None:
        p = Path(config)
        candidates: list[Path] = [p]
        if p.suffix == "":
            candidates.append(p.with_suffix(".json"))
        # 也允许传入裸名称（或裸文件名），在 configs/ 目录下查找
        candidates.append(profiles_dir / p)
        if p.suffix == "":
            candidates.append((profiles_dir / p).with_suffix(".json"))

        for cand in candidates:
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Config file not found: {str(config)!r} (looked for {', '.join(map(str, candidates))})")

    if default_path.is_file():
        return default_path
    return None


def _unwrap_args_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    # 支持复用 runs/<...>/configs/run.json（其参数存储在 "args" 键下）
    args = cfg.get("args")
    if isinstance(args, dict):
        return args
    return cfg


def select_section(cfg: dict[str, Any], *, section: str) -> dict[str, Any]:
    cfg = _unwrap_args_payload(cfg)
    sect = cfg.get(section)
    if isinstance(sect, dict):
        return sect
    return cfg


def parser_defaults(parser: argparse.ArgumentParser, *, exclude: set[str] | None = None) -> dict[str, object]:
    exclude = set() if exclude is None else set(exclude)
    out: dict[str, object] = {}
    for act in parser._actions:
        dest = getattr(act, "dest", None)
        if not dest or dest in ("help",) or dest in exclude:
            continue
        if getattr(act, "default", argparse.SUPPRESS) is argparse.SUPPRESS:
            continue
        out[str(dest)] = _json_compatible(getattr(act, "default"))
    return out


def _action_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for act in parser._actions:
        dest = getattr(act, "dest", None)
        if not dest or dest == "help":
            continue
        actions[str(dest)] = act
    return actions


def _coerce_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p for p in value.split() if p]
        return [*parts]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_action_value(action: argparse.Action, value: object) -> object:
    if value is None:
        return None

    # nargs="*" / "+" 风格的参数
    nargs = getattr(action, "nargs", None)
    if nargs in ("*", "+") or isinstance(value, list):
        items = _coerce_list(value)
        if action.type is None:
            return items
        return [action.type(v) if v is not None else None for v in items]

    if action.type is None:
        return value
    return action.type(value)


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    cfg: dict[str, Any],
    *,
    strict: bool = True,
    allow_unknown_prefixes: tuple[str, ...] = ("_",),
) -> None:
    actions = _action_by_dest(parser)

    unknown = []
    coerced: dict[str, object] = {}
    for k, v in cfg.items():
        key = str(k)
        if any(key.startswith(p) for p in allow_unknown_prefixes):
            continue
        act = actions.get(key)
        if act is None:
            unknown.append(key)
            continue
        coerced[key] = _coerce_action_value(act, v)

    if strict and unknown:
        unknown_s = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unknown config keys for this command: {unknown_s}")

    if coerced:
        parser.set_defaults(**coerced)
