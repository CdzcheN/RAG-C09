#!/usr/bin/env python3
"""src.detection 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法:
  # exp_id 默认从 --features 文件名派生，保证与 features/<exp_id> 同名（契约 §3）
  python -m src.detection.cli --config configs/detect.yaml --seed 13 \\
      --features results/features/w2-baseline-pipeline-13.parquet \\
      [--classifier logreg] [--feature-set main|random|self_confidence|overlap_only] \\
      [--drop-group semantic] [--limit 30] [--dry-run]

产出:
  results/metrics/<exp_id>.json            契约 §2.4 的 MetricRecord + 分组指标 + CV 指标
  results/metrics/<exp_id>-calibration.json 各校准方法的阈值指标（Abl-3）
  results/scores/<exp_id>.jsonl            逐样本分数（evaluation 的配对检验与出图数据源）
  results/models/<exp_id>.joblib           已训练判别器

退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import random
import sys
from typing import Any

from ..common import io
from ..common.cli_utils import add_common_args, make_exp_id
from ..common.logging_utils import get_logger
from ..common.schema import FeatureRow, MetricRecord
from ..evaluation.grouped import group_sizes, grouped_metrics
from ..evaluation.metrics import latency_stats, score_metrics
from .calibrate import CALIBRATION_METHODS, best_threshold, calibrate
from .predict import predict
from .train import train_and_evaluate

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "detect.yaml"
COMPONENT = "detect"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="训练判别器并评估")
    add_common_args(ap)
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "gbdt", "mlp"], help="判别器类型")
    ap.add_argument("--features", default=None, help="特征表（缺省按 exp_id 推断 results/features/<exp_id>.*）")
    ap.add_argument("--fit-split", default="data/processed/split_dev_fit.jsonl",
                    help="dev-fit 划分文件（决定训练侧；其余为报告侧）")
    ap.add_argument("--feature-set", default="main",
                    help="main（三组一致性特征）或 baselines 中的名称（random/self_confidence/overlap_only）")
    ap.add_argument("--drop-group", action="append", default=None,
                    help="Abl-2：去掉某个特征组（semantic/overlap/conflict），可重复")
    ap.add_argument("--no-save-model", action="store_true", help="不落盘判别器（仅出指标）")
    return ap


def merge_config(config_path: str) -> dict[str, Any]:
    cfg = io.load_config(ROOT / "configs" / "default.yaml")
    cfg["models"] = io.load_config(ROOT / "configs" / "models.yaml")
    cfg.update(io.load_config(ROOT / "configs" / "detect.yaml"))
    path = pathlib.Path(config_path)
    if path.resolve() != DEFAULT_CONFIG.resolve():
        cfg.update(io.load_config(path))
    return cfg


def features_path(features_arg: str | None, exp_id: str) -> pathlib.Path:
    if features_arg:
        return pathlib.Path(features_arg)
    for suffix in (".parquet", ".csv"):
        candidate = ROOT / "results" / "features" / f"{exp_id}{suffix}"
        if candidate.exists():
            return candidate
    return ROOT / "results" / "features" / f"{exp_id}.parquet"


def feature_columns(cfg: dict[str, Any], feature_set: str, drop_groups: list[str] | None) -> list[str]:
    """按 `--feature-set` 与 `--drop-group` 决定特征列（配置驱动，不在代码里写列名）。"""
    groups = dict(cfg.get("feature_groups") or {})
    if feature_set != "main":
        baselines = dict(cfg.get("baselines") or {})
        if feature_set not in baselines:
            raise ValueError(f"未知特征集：{feature_set!r}（可选 main 或 {sorted(baselines)}）")
        return [str(c) for c in baselines[feature_set]]
    dropped = set(drop_groups or [])
    unknown = dropped - set(groups)
    if unknown:
        raise ValueError(f"未知特征组：{sorted(unknown)}（可选 {sorted(groups)}）")
    columns: list[str] = []
    for name in sorted(groups):
        if name in dropped:
            continue
        columns.extend(str(c) for c in groups[name])
    return columns


def _load_fit_keys(fit_split: str) -> set[str]:
    path = pathlib.Path(fit_split)
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in io.iter_jsonl(path)}


def _random_scores(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(count)]


def _latency_mean(predictions_file: pathlib.Path) -> float:
    if not predictions_file.exists():
        return float("nan")
    values = [float(row.get("latency_ms", float("nan"))) for row in io.iter_jsonl(predictions_file)]
    return float(latency_stats(values)["latency_ms_mean"])


def _short_model_name(cfg: dict[str, Any], kind: str, feature_set: str) -> str:
    base = str((cfg.get("generation") or {}).get("model", "generator")).split("/")[-1].lower()
    return f"{base}+{kind}" if feature_set == "main" else f"{base}+baseline_{feature_set}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = merge_config(args.config)
        columns = feature_columns(cfg, args.feature_set, args.drop_group)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 配置读取失败：{exc!r}", file=sys.stderr)
        return 1

    seed = int(args.seed if args.seed is not None else (cfg.get("seeds") or [13])[0])
    exp_id = args.exp_id or make_exp_id(str(cfg.get("exp_prefix", "exp")), COMPONENT, seed)
    log = get_logger(f"src.{COMPONENT}", exp_id=exp_id)
    out_root = pathlib.Path(args.out) if args.out else ROOT / "results"
    feat_path = features_path(args.features, exp_id)
    log.info("component=%s exp_id=%s seed=%s classifier=%s feature_set=%s cols=%s features=%s",
             COMPONENT, exp_id, seed, args.classifier, args.feature_set, columns, feat_path)

    if args.dry_run:
        log.info("dry-run：配置与参数检查通过（cv_folds=%s, threshold=%s, 校准=%s）",
                 cfg.get("cv_folds"), cfg.get("threshold"), cfg.get("calibration"))
        return 0

    if not feat_path.exists():
        log.error("特征表不存在：%s（先跑 python -m src.features.cli）", feat_path)
        return 1

    try:
        rows = [FeatureRow.from_dict(row) for row in io.read_table(feat_path)]
        if args.limit is not None:
            rows = rows[:args.limit]
        if not rows:
            log.error("特征表为空：%s", feat_path)
            return 2

        fit_keys = _load_fit_keys(args.fit_split)
        if fit_keys:
            train_rows = [r for r in rows if r.sample_id in fit_keys]
            test_rows = [r for r in rows if r.sample_id not in fit_keys]
        else:
            log.warning("未找到 %s，训练侧与报告侧都使用全部样本（正式实验前须先跑 src.datasets.cli）",
                        args.fit_split)
            train_rows = test_rows = rows
        if not train_rows or not test_rows or len({r.is_hallucination for r in train_rows}) < 2:
            log.error("划分后样本不足或标签单一：train=%d, test=%d", len(train_rows), len(test_rows))
            return 2
        log.info("划分：train(dev-fit)=%d, test(报告侧)=%d；分组样本量=%s",
                 len(train_rows), len(test_rows), group_sizes(test_rows))

        cv_folds = int(cfg.get("cv_folds", 5))
        threshold = float(cfg.get("threshold", 0.5))
        params = dict((cfg.get("detector_params") or {}).get(args.classifier) or {})

        if args.feature_set == "random":  # B0 下界参照：不训练
            train_scores = _random_scores(len(train_rows), seed)
            test_scores = _random_scores(len(test_rows), seed + 1)
            cv_metrics: dict[str, Any] = {}
            model = None
        elif args.feature_set == "self_confidence":  # B1：1 − 自评置信度 作为幻觉分数（方法复现）
            train_scores = [1.0 - float(r.baseline_confidence) for r in train_rows]
            test_scores = [1.0 - float(r.baseline_confidence) for r in test_rows]
            cv_metrics = {}
            model = None
        else:
            trained = train_and_evaluate(train_rows, columns, args.classifier, seed,
                                         cv_folds=cv_folds, params=params)
            model = trained["model"]
            cv_metrics = trained["cv_metrics"]
            train_scores = [float(s) for s in trained["oof_scores"]]
            records = predict(exp_id, model, test_rows, columns,
                              config_hash=io.config_hash(cfg),
                              model_name=_short_model_name(cfg, args.classifier, args.feature_set))
            test_scores = [float(r["hallucination_prob"]) for r in records]

        primary = score_metrics([r.is_hallucination for r in test_rows], test_scores, threshold=threshold)
        grouped = grouped_metrics(test_rows, test_scores)
        log.info("报告侧指标：AUC=%.4f, PR-AUC=%.4f, F1(macro)=%.4f, n=%d",
                 primary["auc"], primary["pr_auc"], primary["f1_macro"], primary["n_samples"])

        calibration_report: dict[str, Any] = {}
        for method in (cfg.get("calibration") or ["none"]):
            if method not in CALIBRATION_METHODS:
                log.warning("忽略未知校准方法：%s", method)
                continue
            mapper = calibrate(train_scores, [r.is_hallucination for r in train_rows], method=method)
            calibrated = mapper(test_scores)
            entry = score_metrics([r.is_hallucination for r in test_rows], calibrated, threshold=threshold)
            entry["best_threshold"] = best_threshold(calibrated, [r.is_hallucination for r in test_rows])
            entry["best_threshold_metrics"] = score_metrics(
                [r.is_hallucination for r in test_rows], calibrated, threshold=entry["best_threshold"])
            calibration_report[method] = entry

        record = MetricRecord(
            exp_id=exp_id, stage="W4", seed=seed,
            model=_short_model_name(cfg, args.classifier, args.feature_set),
            split="challenge", n_samples=int(primary["n_samples"]),
            auc=float(primary["auc"]), pr_auc=float(primary["pr_auc"]),
            precision=float(primary["precision"]), recall=float(primary["recall"]),
            f1_macro=float(primary["f1_macro"]), f1_micro=float(primary["f1_micro"]),
            confusion=dict(primary["confusion"]), threshold=threshold,
            latency_ms_mean=_latency_mean(ROOT / "results" / "predictions" / f"{exp_id}.jsonl"),
            config_hash=io.config_hash(cfg), env_report="results/env_report.json",
            timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
            notes=f"feature_set={args.feature_set}; drop_group={args.drop_group or []}; "
                  f"calibration={list(calibration_report)}; cv_folds={cv_folds}; "
                  f"train_n={len(train_rows)}; test_n={len(test_rows)}",
        )
        payload = {**record.to_dict(), "features_file": str(feat_path), "feature_cols": columns,
                   "cv_metrics": cv_metrics, "grouped": grouped,
                   "label_counts": {str(v): sum(1 for r in test_rows if r.is_hallucination == v)
                                    for v in sorted({r.is_hallucination for r in test_rows})},
                   "dataset_revision": {}, "notes_extra": "指标口径见 docs/实验与评估规范.md §1"}
        io.write_json(out_root / "metrics" / f"{exp_id}.json", payload)
        io.write_json(out_root / "metrics" / f"{exp_id}-calibration.json",
                      {"exp_id": exp_id, "seed": seed, "methods": calibration_report,
                       "config_hash": payload["config_hash"]})

        io.write_jsonl(out_root / "scores" / f"{exp_id}.jsonl",
                       [{"sample_id": r.sample_id, "score": float(s), "is_hallucination": int(r.is_hallucination),
                         "challenge_type": r.challenge_type, "method": record.model,
                         "baseline_confidence": float(r.baseline_confidence)}
                        for r, s in zip(test_rows, test_scores)],
                       meta={"exp_id": exp_id, "seed": seed, "config_hash": payload["config_hash"],
                             "method": record.model, "feature_cols": columns,
                             "threshold": threshold, "split": "challenge"})

        if model is not None and not args.no_save_model:
            try:
                import joblib

                models_dir = out_root / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": model, "feature_cols": columns, "exp_id": exp_id,
                             "classifier": args.classifier}, models_dir / f"{exp_id}.joblib")
                log.info("判别器已落盘：%s.joblib", models_dir / exp_id)
            except Exception as exc:  # noqa: BLE001 - 落盘失败不影响指标产物
                log.warning("判别器落盘失败（%r）", exc)
        log.info("指标已落盘：%s（校准对比：%s）", out_root / "metrics" / f"{exp_id}.json",
                 list(calibration_report))
    except (ValueError, FileNotFoundError) as exc:
        log.error("输入错误：%r", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("运行失败：%r", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
