#!/usr/bin/env python3
"""概率校准（Platt / 温度缩放）。

负责人: C
对应文档: docs/实验与评估规范.md §3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def calibrate(scores: Sequence[float], y_true: Sequence[int], method: str = "platt") -> Any:
    """拟合校准器并返回可调用的映射函数。

    状态: 待实现 —— T4.1：Abl-3 的比较项；需出可靠性图
    """
    raise NotImplementedError("T4.1：Abl-3 的比较项；需出可靠性图")


