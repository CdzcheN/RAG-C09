#!/usr/bin/env python3
"""配对显著性检验（配对 t / Wilcoxon）。

负责人: C
对应文档: docs/实验与评估规范.md §4
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def paired_tests(y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """对同一样本上的两种方法做配对 t 检验与 Wilcoxon 符号秩检验。

    状态: 待实现 —— T5.1：配对单位必须是 sample_id；两个 p 值都要报
    """
    raise NotImplementedError("T5.1：配对单位必须是 sample_id；两个 p 值都要报")


