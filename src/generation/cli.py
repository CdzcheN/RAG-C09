#!/usr/bin/env python3
"""src.generation 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法:
  python -m src.generation.cli --config configs/default.yaml --seed 13 [--limit 20] [--dry-run]

产出: `results/predictions/<exp_id>.jsonl`（首行 `_meta`，契约 §4）。
样本输入默认取 `data/processed/challenge_set.jsonl`，不存在时回退 `normal_set.jsonl`。

退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Any

from ..common import io, validate
from ..common.cli_utils import add_common_args, make_exp_id
from ..common.logging_utils import get_logger
from ..common.schema import SCHEMA_VERSION, RAGSample
from .pipeline import COMPONENT, generation_settings, run_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="跑 RAG 管道并产出 predictions")
    add_common_args(ap)
    ap.add_argument("--samples", default=None, help="样本文件（challenge_set / normal_set / split_*）")
    return ap


def merge_config(config_path: str) -> dict[str, Any]:
    """合并 default.yaml、models.yaml 与指定配置（生成参数分散在这几份里）。"""
    cfg = io.load_config(DEFAULT_CONFIG)
    cfg["models"] = io.load_config(ROOT / "configs" / "models.yaml")
    path = pathlib.Path(config_path)
    if path.resolve() != DEFAULT_CONFIG.resolve():
        cfg.update(io.load_config(path))
    return cfg


def _default_samples_path(cfg: dict[str, Any]) -> pathlib.Path:
    processed = ROOT / str((cfg.get("paths") or {}).get("data_processed", "data/processed"))
    for name in ("challenge_set.jsonl", "normal_set.jsonl"):
        candidate = processed / name
        if candidate.exists():
            return candidate
    return processed / "challenge_set.jsonl"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _output_path(out_arg: str | None, exp_id: str) -> pathlib.Path:
    if out_arg:
        path = pathlib.Path(out_arg)
        return path if path.suffix == ".jsonl" else path / f"{exp_id}.jsonl"
    return ROOT / "results" / "predictions" / f"{exp_id}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = merge_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 配置读取失败：{exc!r}", file=sys.stderr)
        return 1

    seed = int(args.seed if args.seed is not None else (cfg.get("seeds") or [13])[0])
    exp_id = args.exp_id or make_exp_id(str(cfg.get("exp_prefix", "exp")), COMPONENT, seed)
    log = get_logger(f"src.{COMPONENT}", exp_id=exp_id)
    samples_path = pathlib.Path(args.samples) if args.samples else _default_samples_path(cfg)
    log.info("component=%s exp_id=%s seed=%s limit=%s dry_run=%s samples=%s",
             COMPONENT, exp_id, seed, args.limit, args.dry_run, samples_path)

    if args.dry_run:
        settings = generation_settings(cfg)
        log.info("dry-run：配置与参数检查通过（model=%s, dtype=%s, do_sample=%s, max_new_tokens=%s）",
                 settings["model_id"], settings["dtype"], settings["do_sample"],
                 settings["max_new_tokens"])
        return 0

    if not samples_path.exists():
        log.error("样本文件不存在：%s（先跑 python -m src.datasets.cli）", samples_path)
        return 1

    try:
        samples = [RAGSample.from_dict(row) for row in io.iter_jsonl(samples_path)]
        if not samples:
            log.error("样本文件为空：%s", samples_path)
            return 2
        predictions = run_pipeline(samples, cfg, seed, limit=args.limit)
        out_path = _output_path(args.out, exp_id)
        meta = io.build_meta(exp_id, seed, cfg, model=generation_settings(cfg)["model_id"],
                            dataset_revision={"samples_file": str(samples_path)},
                            extra={"schema_version": SCHEMA_VERSION, "git_commit": _git_commit(),
                                   "n_samples": len(predictions),
                                   "decode": predictions[0].decode if predictions else {}})
        io.write_jsonl(out_path, [p.to_dict() for p in predictions], meta=meta)
        log.info("预测已落盘：%s（%d 条）", out_path.relative_to(ROOT), len(predictions))
    except (ValueError, FileNotFoundError) as exc:
        log.error("输入错误：%r", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("运行失败：%r", exc)
        return 2

    problems = validate.validate_file("predictions", out_path)
    if problems:
        for problem in problems[:20]:
            log.error("产物校验：%s", problem)
        return 3
    log.info("产物校验通过（predictions）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
