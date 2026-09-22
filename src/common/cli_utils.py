#!/usr/bin/env python3
"""CLI 公共参数与启动助手（docs/编码与协作规范.md §2）。

统一参数：--config --seed --out --limit --dry-run --exp-id
exp_id 命名：{exp_prefix}-{component}-{seed}（见 docs/接口契约.md §3）
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

from .io import load_config
from .logging_utils import get_logger


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """为子命令添加统一参数。"""
    ap.add_argument("--config", default="configs/default.yaml", help="配置文件路径（YAML）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（缺省取配置中的首个种子）")
    ap.add_argument("--out", default=None, help="输出目录/文件（覆盖默认路径）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（冒烟用）")
    ap.add_argument("--dry-run", action="store_true", help="只校验配置与输入，不执行重计算")
    ap.add_argument("--exp-id", default=None, help="运行 ID；缺省时按命名规则自动生成")


def make_exp_id(prefix: str, component: str, seed: int) -> str:
    """按 docs/接口契约.md §3 的规则生成 exp_id。"""
    return f"{prefix}-{component}-{seed}"


def bootstrap(args: argparse.Namespace, component: str) -> tuple[logging.Logger, dict[str, Any], str]:
    """载入配置、确定种子与 exp_id、初始化日志；返回 (log, cfg, exp_id)。"""
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seeds", [13])[0])
    exp_id = args.exp_id or make_exp_id(str(cfg.get("exp_prefix", "exp")), component, seed)
    log = get_logger(f"src.{component}", exp_id=exp_id)
    log.info("component=%s exp_id=%s seed=%s limit=%s dry_run=%s",
             component, exp_id, seed, args.limit, args.dry_run)
    return log, cfg, exp_id


def todo(component: str, task: str) -> None:
    """未实现时的统一出口：明确报错，而不是静默返回成功。"""
    raise NotImplementedError(
        f"[{component}] 尚未实现：{task}（见 docs/编码与协作规范.md §2 与 docs/项目启动与实施指南.md §4.3 的 WBS）"
    )
