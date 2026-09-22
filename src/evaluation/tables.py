#!/usr/bin/env python3
"""结果表导出（markdown / CSV）。

负责人: C
对应文档: docs/实验与评估规范.md §5
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def to_markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """把指标汇总渲染为 markdown 表，供技术报告直接粘贴。

    状态: 待实现 —— T6.2：与图表同源，避免手抄数字出错
    """
    raise NotImplementedError("T6.2：与图表同源，避免手抄数字出错")


