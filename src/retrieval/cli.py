#!/usr/bin/env python3
"""src.retrieval 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法: python -m src.retrieval.cli --config configs/default.yaml --seed 13 [--limit 20] [--dry-run]
退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse

from ..common.cli_utils import add_common_args, bootstrap, todo


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="构建检索索引并评估 Recall@k")
    add_common_args(ap)
    ap.add_argument("--top-k", type=int, default=None, help="检索条数，缺省取配置值")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log, cfg, exp_id = bootstrap(args, component="retrieval")
    if args.dry_run:
        log.info("dry-run：配置与参数检查通过，未执行计算")
        return 0
    todo("retrieval", "T1.2：建索引并输出 Recall@k 报告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
