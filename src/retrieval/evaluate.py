#!/usr/bin/env python3
"""检索质量评估：Recall@k。

负责人: A
对应文档: docs/数据构造规范.md §5
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def recall_at_k(retrieved: Sequence[Sequence[str]], gold: Sequence[set[str]], k: int) -> float:
    """计算 Recall@k（gold 证据是否出现在前 k 条）。

    状态: 待实现 —— T1.2：在 dev 抽样 200 问上给出 Recall@5 报告
    """
    raise NotImplementedError("T1.2：在 dev 抽样 200 问上给出 Recall@5 报告")


