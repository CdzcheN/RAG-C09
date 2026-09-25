#!/usr/bin/env python3
"""src.evaluation 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法:
  python -m src.evaluation.cli --config configs/detect.yaml \\
      --exp-ids w4-consistency-logreg-13 w4-consistency-logreg-42 w4-consistency-logreg-2024 \\
      [--baseline-exp-id w4-baselineselfconf-logreg-13] [--no-figures] [--dry-run]

产出:
  results/metrics/summary.csv   每行 = (方法, 集合)，含 auc_mean / auc_std / p_value_t / p_value_wilcoxon
  results/metrics/summary.json  配对检验与汇总明细
  results/metrics/summary.md    markdown 表格（可直接粘进技术报告）
  results/figures/*.png         六张必备图（由 metrics/ 与 scores/ 数据重绘）

退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

from ..common import io
from ..common.cli_utils import add_common_args, make_exp_id
from ..common.logging_utils import get_logger
from .grouped import grouped_metrics
from .metrics import score_metrics
from .plots import (plot_ablation, plot_confusion, plot_feature_distribution, plot_grouped_bars,
                    plot_reliability, plot_roc_pr)
from .significance import paired_tests
from .tables import aggregate, to_markdown_table

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPONENT = "eval"
SUMMARY_COLUMNS = ("method", "split", "n_runs", "n_samples", "auc", "auc_std", "pr_auc", "pr_auc_std",
                   "precision", "recall", "f1_macro", "f1_macro_std")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="分组评估、显著性检验与出图")
    add_common_args(ap)
    ap.add_argument("--exp-ids", nargs="*", default=None, help="要评估的运行 ID（可多个）")
    ap.add_argument("--baseline-exp-id", default=None, help="配对检验的对照运行（缺省自动找 B1 自评置信度）")
    ap.add_argument("--no-figures", action="store_true", help="只出表与检验，不绘图")
    ap.add_argument("--scores-dir", default=None,
                    help="逐样本分数目录（默认 results/scores；detection 用了 --out 时指向同一根）")
    return ap


def _metrics_path(token: str) -> pathlib.Path:
    path = pathlib.Path(token)
    if path.suffix == ".json":
        return path
    return ROOT / "results" / "metrics" / f"{token}.json"


def _load_metrics(tokens: list[str]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for token in tokens:
        path = _metrics_path(token)
        if not path.exists():
            raise FileNotFoundError(f"指标文件不存在：{path}")
        payload = io.read_json(path)
        loaded[str(payload.get("exp_id") or path.stem)] = payload
    return loaded


def _scores_dir(token: str | None) -> pathlib.Path:
    return pathlib.Path(token) if token else ROOT / "results" / "scores"


def _scores_map(exp_id: str, scores_dir: pathlib.Path | None = None) -> dict[str, float]:
    path = (scores_dir or _scores_dir(None)) / f"{exp_id}.jsonl"
    if not path.exists():
        return {}
    return {str(row["sample_id"]): float(row["score"]) for row in io.iter_jsonl(path)}


def _find_baseline(metrics: dict[str, dict[str, Any]]) -> str | None:
    for exp_id, payload in metrics.items():
        if "baseline_self_confidence" in str(payload.get("model", "")):
            return exp_id
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed = int(args.seed if args.seed is not None else 13)
    exp_id = args.exp_id or make_exp_id("summary", COMPONENT, seed)
    log = get_logger(f"src.{COMPONENT}", exp_id=exp_id)
    log.info("component=%s exp_id=%s exp_ids=%s baseline=%s dry_run=%s",
             COMPONENT, exp_id, args.exp_ids, args.baseline_exp_id, args.dry_run)

    tokens = list(args.exp_ids or [])
    try:
        cfg = io.load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 配置读取失败：{exc!r}", file=sys.stderr)
        return 1
    alpha = float((cfg.get("significance") or {}).get("alpha", 0.05))
    out_dir = pathlib.Path(args.out) if args.out else ROOT / "results" / "metrics"
    scores_dir = _scores_dir(args.scores_dir)
    if args.dry_run:  # 与其它 CLI 一致：dry-run 只校验配置与打印参数，不要求输入齐备
        if not tokens:
            log.warning("dry-run：未提供 --exp-ids（真实运行必需）")
        log.info("dry-run：参数检查通过（%d 个运行，figures=%s）", len(tokens), not args.no_figures)
        return 0
    if not tokens:
        log.error("必须给出 --exp-ids（例如 w4-consistency-logreg-13）")
        return 1

    try:
        metrics = _load_metrics(tokens)
        records = [{"exp_id": key, **payload} for key, payload in metrics.items()]
        summary_rows = aggregate(records, ["method", "split"],
                                ["auc", "pr_auc", "precision", "recall", "f1_macro"])
        for row in summary_rows:  # 样本量取各次运行的众数（同一集合应一致）
            sizes = [int(p["n_samples"]) for p in records
                     if p.get("method") == row["method"] and p.get("split") == row["split"]]
            row["n_samples"] = max(sizes) if sizes else None
        summary_path = io.write_table(out_dir / "summary.csv", summary_rows,
                                      columns=list(SUMMARY_COLUMNS))
        log.info("汇总表已落盘：%s（%d 行）", summary_path.name, len(summary_rows))

        # 配对检验：M 与 B1 在同一批 sample_id 上比较（配对单位 = sample_id）
        baseline_id = args.baseline_exp_id or _find_baseline(metrics)
        primary_id = next((key for key, p in metrics.items() if "baseline" not in str(p.get("model", ""))),
                          list(metrics)[0])
        significance: dict[str, Any] = {}
        if baseline_id and baseline_id != primary_id:
            left = _scores_map(primary_id, scores_dir)
            right = _scores_map(baseline_id, scores_dir)
            shared = sorted(set(left) & set(right))
            if shared:
                scores_path = scores_dir / f"{primary_id}.jsonl"
                labels = {str(row["sample_id"]): int(row["is_hallucination"])
                          for row in io.iter_jsonl(scores_path)}
                significance = paired_tests([labels[s] for s in shared], [left[s] for s in shared],
                                            [right[s] for s in shared], alpha=alpha,
                                            family_size=int(len(metrics)), label_a=primary_id,
                                            label_b=baseline_id)
                log.info("配对检验（%d 对）：ΔAUC=%.4f, p_t=%.4g, p_wilcoxon=%.4g",
                         significance["n_pairs"], significance["delta_auc"],
                         significance["ttest"]["p_value"], significance["wilcoxon"]["p_value"])
            else:
                log.warning("M 与对照没有共同的 sample_id，跳过配对检验")
        else:
            log.warning("未找到可用的对照运行（--baseline-exp-id），跳过配对检验")

        markdown = to_markdown_table(summary_rows, columns=["method", "n_runs", "n_samples", "auc",
                                                            "auc_std", "pr_auc", "f1_macro"])
        io.write_json(out_dir / "summary.json",
                      {"exp_id": exp_id, "runs": list(metrics), "summary": summary_rows,
                       "significance": significance, "alpha": alpha,
                       "note": "汇总口径：多 seed 的 mean±std；配对单位 = sample_id"})
        core = {key: significance[key] for key in
                ("label_a", "label_b", "n_pairs", "auc_a", "auc_b", "delta_auc", "alpha",
                 "significant_t", "significant_wilcoxon", "meets_target")
                if key in significance}
        detail = {key: significance[key] for key in ("ttest", "wilcoxon", "holm_ttest", "holm_wilcoxon")
                  if key in significance}
        atomic_write_text = io.atomic_write_text
        atomic_write_text(out_dir / "summary.md",
                          f"# 结果汇总（exp_id: {exp_id}）\n\n{markdown}\n\n"
                          "## 配对检验（M vs 对照，配对单位 = sample_id）\n\n"
                          f"{to_markdown_table([core]) if core else '(未执行：缺少对照运行或共同样本)'}\n\n"
                          f"p 值明细：`{detail}`\n")
    except (ValueError, FileNotFoundError) as exc:
        log.error("输入错误：%r", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("运行失败：%r", exc)
        return 2

    if args.no_figures:
        log.info("按 --no-figures 跳过绘图")
        return 0

    try:
        figures_dir = (pathlib.Path(args.out) / "figures") if args.out else (ROOT / "results" / "figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        primary = metrics[primary_id]
        saved: list[str] = []
        if primary.get("grouped"):
            saved.append(plot_grouped_bars(primary["grouped"], str(figures_dir / f"{primary_id}-grouped.png")))
        primary_scores = _scores_map(primary_id, scores_dir)
        order: list[str] = []
        if primary_scores:
            order = [str(row["sample_id"]) for row in io.iter_jsonl(scores_dir / f"{primary_id}.jsonl")]
        curves: dict[str, list[float]] = {}
        for key in metrics:  # 各方法的分数按同一 sample_id 顺序对齐，避免曲线错配
            mapping = _scores_map(key, scores_dir)
            if mapping and order and all(sample_id in mapping for sample_id in order):
                curves[key] = [mapping[sample_id] for sample_id in order]
        if curves and order:
            labels = [int(row["is_hallucination"]) for row in
                      io.iter_jsonl(scores_dir / f"{primary_id}.jsonl")]
            saved.append(plot_roc_pr(labels, curves, str(figures_dir / f"{primary_id}-roc-pr.png")))
        if primary.get("confusion"):
            saved.append(plot_confusion(primary["confusion"], str(figures_dir / f"{primary_id}-confusion.png")))
        features_file = primary.get("features_file")
        if features_file and pathlib.Path(features_file).exists():
            saved.append(plot_feature_distribution(io.read_table(features_file),
                                                   str(figures_dir / f"{primary_id}-feature-dist.png")))
        ablation = {key: {"auc": payload.get("auc", float("nan")),
                          "auc_std": 0.0, "n_samples": payload.get("n_samples")}
                    for key, payload in metrics.items()}
        if ablation:
            saved.append(plot_ablation(ablation, str(figures_dir / f"{primary_id}-ablation.png")))
        scores = _scores_map(primary_id, scores_dir)
        if scores:
            score_path = scores_dir / f"{primary_id}.jsonl"
            rows = list(io.iter_jsonl(score_path))
            saved.append(plot_reliability([float(r["score"]) for r in rows],
                                          [int(r["is_hallucination"]) for r in rows],
                                          str(figures_dir / f"{primary_id}-reliability.png")))
        log.info("已重绘 %d 张图 → results/figures/", len(saved))
    except Exception as exc:  # noqa: BLE001 - 绘图失败不掩盖已落盘的表格产物
        log.exception("绘图失败：%r", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
