#!/usr/bin/env python3
"""src.detection 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法: python -m src.detection.cli --config configs/detect.yaml --seed 13 [--limit 20] [--dry-run]
退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse

from ..common.cli_utils import add_common_args, bootstrap, todo


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="训练判别器并评估")
    add_common_args(ap)
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "gbdt", "mlp"], help="判别器类型")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log, cfg, exp_id = bootstrap(args, component="detect")
    if args.dry_run:
        log.info("dry-run：配置与参数检查通过，未执行计算")
        return 0
    todo("detect", "T4.1：训练 + 5 折 CV + 校准曲线")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
