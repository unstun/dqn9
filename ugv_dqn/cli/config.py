"""生成合并的 train+infer JSON 配置模板。

通过反射 train.py 和 infer.py 的 argparse 定义，生成
包含所有可用参数及其默认值的单一 JSON。
用法：python config.py [--out configs/template.json | --stdout]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from ugv_dqn.config_io import parser_defaults


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Generate a combined train+infer JSON config template.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("configs/template.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--stdout",
        action="store_true",
        help="Print config JSON to stdout instead of writing a file.",
    )
    args = ap.parse_args(argv)

    # 延迟导入以保持轻量。
    # 部分依赖（如 gym）在导入时可能输出废弃警告；保持 JSON 输出干净。
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ugv_dqn.cli.infer import build_parser as build_infer_parser
        from ugv_dqn.cli.train import build_parser as build_train_parser

    train_parser = build_train_parser()
    infer_parser = build_infer_parser()

    cfg = {
        "train": parser_defaults(train_parser, exclude={"config", "profile"}),
        "infer": parser_defaults(infer_parser, exclude={"config", "profile"}),
    }
    text = json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    if bool(args.stdout):
        sys.stdout.write(text)
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
