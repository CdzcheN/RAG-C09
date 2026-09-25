#!/usr/bin/env python3
"""src.features 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法:
  # exp_id 默认从 --pred 文件名派生，保证与 predictions/<exp_id>.jsonl 同名（契约 §3）
  python -m src.features.cli --config configs/default.yaml --seed 13 \\
      --pred results/predictions/w2-baseline-pipeline-13.jsonl [--limit 20] [--dry-run]

产出: `results/features/<exp_id>.parquet`（未安装 pyarrow 时退化为同目录 `.csv` 并记日志）；
      `results/metrics/<exp_id>-features.json` 记录标签回填统计（undetermined 数量等）。

退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

from ..common import io, validate
from ..common.cli_utils import add_common_args, make_exp_id
from ..common.logging_utils import get_logger
from ..common.schema import FEATURE_COLUMNS, Prediction, RAGSample
from .build_features import build_feature_table
from .entailment import DEFAULT_NLI_ID, load_nli

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
COMPONENT = "features"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="提取一致性特征并落盘 parquet")
    add_common_args(ap)
    ap.add_argument("--pred", default=None, help="predictions 文件（缺省按 exp_id 推断）")
    ap.add_argument("--samples", default=None, help="样本文件（缺省用 data/processed/challenge_set.jsonl）")
    ap.add_argument("--no-nli", action="store_true", help="跳过 NLI 模型（语义/冲突特征记 NaN，仅用于链路自检）")
    return ap


def merge_config(config_path: str) -> dict[str, Any]:
    cfg = io.load_config(DEFAULT_CONFIG)
    cfg["models"] = io.load_config(ROOT / "configs" / "models.yaml")
    path = pathlib.Path(config_path)
    if path.resolve() != DEFAULT_CONFIG.resolve():
        cfg.update(io.load_config(path))
    return cfg


def _predictions_path(pred_arg: str | None, exp_id: str) -> pathlib.Path:
    if pred_arg:
        return pathlib.Path(pred_arg)
    return ROOT / "results" / "predictions" / f"{exp_id}.jsonl"


def _samples_path(samples_arg: str | None, cfg: dict[str, Any]) -> pathlib.Path:
    if samples_arg:
        return pathlib.Path(samples_arg)
    processed = ROOT / str((cfg.get("paths") or {}).get("data_processed", "data/processed"))
    for name in ("challenge_set.jsonl", "normal_set.jsonl"):
        if (processed / name).exists():
            return processed / name
    return processed / "challenge_set.jsonl"


def _features_path(exp_id: str) -> pathlib.Path:
    return ROOT / "results" / "features" / f"{exp_id}.parquet"


def _rel(path: pathlib.Path) -> str:
    """相对仓库根的展示用路径（`--out` 指向仓库外时退化为绝对路径）。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = merge_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 配置读取失败：{exc!r}", file=sys.stderr)
        return 1

    seed = int(args.seed if args.seed is not None else (cfg.get("seeds") or [13])[0])
    exp_id = args.exp_id or (pathlib.Path(args.pred).stem if args.pred
                             else make_exp_id(str(cfg.get("exp_prefix", "exp")), COMPONENT, seed))
    log = get_logger(f"src.{COMPONENT}", exp_id=exp_id)
    pred_path = _predictions_path(args.pred, exp_id)
    samples_path = _samples_path(args.samples, cfg)
    log.info("component=%s exp_id=%s seed=%s limit=%s dry_run=%s pred=%s samples=%s",
             COMPONENT, exp_id, seed, args.limit, args.dry_run, pred_path, samples_path)

    if args.dry_run:
        log.info("dry-run：配置与参数检查通过（特征列 %d 个，NLI=%s）",
                 len(FEATURE_COLUMNS), "关闭" if args.no_nli else DEFAULT_NLI_ID)
        return 0

    for path in (pred_path, samples_path):
        if not path.exists():
            log.error("输入文件不存在：%s", path)
            return 1

    try:
        predictions = [Prediction.from_dict(row) for row in io.iter_jsonl(pred_path)]
        if args.limit is not None:
            predictions = predictions[:args.limit]
        samples = [RAGSample.from_dict(row) for row in io.iter_jsonl(samples_path)]
        if not predictions or not samples:
            log.error("输入为空（predictions=%d, samples=%d）", len(predictions), len(samples))
            return 2

        nli_cfg = dict((cfg.get("models") or {}).get("nli") or {})
        nli = None if args.no_nli else load_nli(str(nli_cfg.get("id", DEFAULT_NLI_ID)),
                                                str(nli_cfg.get("device", "cuda")))
        stats: dict[str, Any] = {}
        rows = build_feature_table(predictions, samples, nli, cfg, stats=stats)
        if not rows:
            log.error("特征表为空（统计：%s）", stats)
            return 2

        out_path = pathlib.Path(args.out) if args.out else _features_path(exp_id)
        written = io.write_table(out_path, [row.to_dict() for row in rows], columns=list(FEATURE_COLUMNS))
        stats_dir = out_path.parent if args.out else ROOT / "results" / "metrics"
        stats_report = {**stats, "exp_id": exp_id, "seed": seed, "config_hash": io.config_hash(cfg),
                        "samples_file": str(samples_path), "predictions_file": str(pred_path),
                        "features_file": str(written), "nli": "none" if args.no_nli else DEFAULT_NLI_ID}
        io.write_json(stats_dir / f"{exp_id}-features.json", stats_report)
        log.info("特征表已落盘：%s（%d 行；undetermined=%s）",
                 _rel(written), len(rows), stats.get("n_undetermined"))
    except (ValueError, FileNotFoundError) as exc:
        log.error("输入错误：%r", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("运行失败：%r", exc)
        return 2

    problems = validate.validate_file("features", written)
    if problems:
        for problem in problems[:20]:
            log.error("产物校验：%s", problem)
        return 3
    log.info("产物校验通过（features）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
