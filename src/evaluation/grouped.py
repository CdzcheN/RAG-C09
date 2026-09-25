#!/usr/bin/env python3
"""按幻觉类型分组评估。

负责人: C
对应文档: docs/实验与评估规范.md §1、§7
状态: 已实现（T4.2）

分组结论是本课题的核心（"三类难度不等，定位难度来源"），因此每组必须报告样本量与
正/负样本数（`score_metrics` 已含 `n_samples` / `n_positive` / `n_negative`）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import FeatureRow
from .metrics import score_metrics

LOG = get_logger(__name__)

ALL = "all"


def grouped_metrics(features: Sequence[FeatureRow], scores: Sequence[float],
                    group_col: str = "challenge_type") -> dict[str, dict[str, Any]]:
    """对每个 `group_col` 取值分别计算指标（含混淆矩阵与宏/微 F1），并附整体 `all`。

    返回 `{组名: 指标字典,..., "all": 指标字典}`。正类固定为幻觉（`is_hallucination=1`）。
    """
    if len(features) != len(scores):
        raise ValueError("features 与 scores 长度不一致")

    buckets: dict[str, tuple[list[int], list[float]]] = {}
    for row, score in zip(features, scores):
        key = str(getattr(row, group_col, "unknown"))
        labels, values = buckets.setdefault(key, ([], []))
        labels.append(int(row.is_hallucination))
        values.append(float(score))

    result: dict[str, dict[str, Any]] = {}
    for key in sorted(buckets):
        labels, values = buckets[key]
        result[key] = score_metrics(labels, values)
        LOG.debug("分组 %s：n=%d, AUC=%s", key, result[key]["n_samples"], result[key]["auc"])

    all_labels = [int(row.is_hallucination) for row in features]
    result[ALL] = score_metrics(all_labels, [float(s) for s in scores])
    return result


def group_sizes(features: Sequence[FeatureRow], group_col: str = "challenge_type") -> Mapping[str, int]:
    """各组样本量（报告中"必须 report 每组样本量"的直接来源）。"""
    sizes: dict[str, int] = {}
    for row in features:
        key = str(getattr(row, group_col, "unknown"))
        sizes[key] = sizes.get(key, 0) + 1
    return sizes
