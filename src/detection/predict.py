#!/usr/bin/env python3
"""推理与产物落盘。

负责人: C
对应文档: docs/接口契约.md §2.4
状态: 已实现（T4.1）

输出逐样本记录 `{sample_id, exp_id, hallucination_prob, is_hallucination, challenge_type, ...}`，
其中元信息字段（`config_hash` / `env_report` / `timestamp` / `model`）由调用方注入，最终与
`evaluation.metrics.score_metrics` 的结果一起组装为契约 §2.4 的 `MetricRecord`。
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import FeatureRow

LOG = get_logger(__name__)

ENV_REPORT_DEFAULT = "results/env_report.json"


def _matrix(features: Sequence[FeatureRow], feature_cols: Sequence[str]) -> list[list[float]]:
    return [[float(getattr(row, col, float("nan"))) for col in feature_cols] for row in features]


def predict(exp_id: str, model: Any, features: Sequence[FeatureRow],
            feature_cols: Sequence[str], *, config_hash: str = "",
            env_report: str = ENV_REPORT_DEFAULT, model_name: str = "") -> list[dict[str, Any]]:
    """输出幻觉概率并组装 MetricRecord 所需字段（含 `config_hash` 与 `env_report`）。

    `model` 为 `detection.train.train_and_evaluate` 返回的已拟合判别器（须有 `predict_proba`）。
    返回的每条记录都带同一份运行元信息，便于逐样本追溯（契约 §4 的可追溯性要求）。
    """
    if not features:
        return []
    matrix = _matrix(features, feature_cols)
    probabilities = [float(p[1]) for p in model.predict_proba(matrix)]
    if len(probabilities) != len(features):
        raise RuntimeError(f"预测条数({len(probabilities)})与特征行数({len(features)})不一致")

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    for row, prob in zip(features, probabilities):
        records.append({
            "sample_id": row.sample_id,
            "exp_id": exp_id,
            "hallucination_prob": prob,
            "is_hallucination": int(row.is_hallucination),
            "challenge_type": row.challenge_type,
            "baseline_confidence": float(row.baseline_confidence),
            "feature_cols": list(feature_cols),
            "model": model_name,
            "config_hash": config_hash,
            "env_report": env_report,
            "timestamp": timestamp,
        })
    LOG.info("推理完成：%d 条（%s）", len(records), exp_id)
    return records


def scores_of(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """取出幻觉分数序列（评估层与显著性检验共用）。"""
    return [float(r["hallucination_prob"]) for r in records]


def labels_of(records: Sequence[Mapping[str, Any]]) -> list[int]:
    """取出标签序列。"""
    return [int(r["is_hallucination"]) for r in records]
