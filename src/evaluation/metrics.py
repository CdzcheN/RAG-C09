#!/usr/bin/env python3
"""指标计算：AUC / PR-AUC / 阈值指标 / 混淆矩阵（docs/实验与评估规范.md §1）。

负责人: C
说明: 正类为幻觉（is_hallucination = 1），必须显式指定 labels，避免标签顺序导致数值反转。
    延迟导入 scikit-learn，使 CLI --help 在依赖未装齐时仍可用。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _sklearn():
    try:
        from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                     precision_score, recall_score, roc_auc_score)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "计算指标需要 scikit-learn，请先执行 python -m pip install -r requirements.txt"
        ) from exc
    return (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix)


def score_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    """给定标签与幻觉分数，返回契约 §2.4 的指标字典。

    单一类别时 AUC 未定义，返回 NaN 而不是抛异常（报告中需标注该情况）。
    """
    (roc_auc_score, average_precision_score, precision_score,
     recall_score, f1_score, confusion_matrix) = _sklearn()

    y_true = [int(v) for v in y_true]
    y_score = [float(v) for v in y_score]
    if len(y_true) != len(y_score):
        raise ValueError("y_true 与 y_score 长度不一致")
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "n_samples": len(y_true),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
        "threshold": float(threshold),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    if len(set(y_true)) < 2:
        result["auc"] = float("nan")
        result["pr_auc"] = float("nan")
    else:
        result["auc"] = float(roc_auc_score(y_true, y_score))
        result["pr_auc"] = float(average_precision_score(y_true, y_score))
    return result


def latency_stats(latencies_ms: Sequence[float]) -> Mapping[str, float]:
    """效率指标：均值与 P95 延迟（效率类指标，要求.md 2.3 条第 1 款）。"""
    values = sorted(float(x) for x in latencies_ms if x == x)  # 过滤 NaN
    if not values:
        return {"latency_ms_mean": float("nan"), "latency_ms_p95": float("nan")}
    idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return {"latency_ms_mean": sum(values) / len(values), "latency_ms_p95": values[idx]}
