#!/usr/bin/env python3
"""按幻觉类型分组评估。

负责人: C
对应文档: docs/实验与评估规范.md §1、§7
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def grouped_metrics(features: Sequence[FeatureRow], scores: Sequence[float], group_col: str = "challenge_type") -> dict[str, dict[str, Any]]:
    """对每个 challenge_type 分别计算指标（含混淆矩阵与宏/微 F1）。

    状态: 待实现 —— T4.2：分组结论是课题结论的核心，必须 report 每组样本量
    """
    raise NotImplementedError("T4.2：分组结论是课题结论的核心，必须 report 每组样本量")


