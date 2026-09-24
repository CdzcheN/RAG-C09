#!/usr/bin/env python3
"""训练与交叉验证。

负责人: C
对应文档: docs/实验与评估规范.md §4
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def train_and_evaluate(features: Sequence[FeatureRow], feature_cols: Sequence[str], kind: str, seed: int, cv_folds: int = 5) -> dict[str, Any]:
    """5 折交叉验证训练并返回指标与模型对象。

    状态: 待实现 —— T4.1：训练集用 dev-fit，报告用 dev-test
    """
    raise NotImplementedError("T4.1：训练集用 dev-fit，报告用 dev-test")


