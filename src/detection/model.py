#!/usr/bin/env python3
"""判别器定义（LogReg / GBDT / 轻量 MLP）。

负责人: C
对应文档: docs/实验与评估规范.md §3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def make_classifier(kind: str = "logreg", seed: int = 13) -> Any:
    """按名称返回判别器实例（必须固定 random_state）。

    状态: 待实现 —— T4.1：Abl-3 需要三种可替换判别器
    """
    raise NotImplementedError("T4.1：Abl-3 需要三种可替换判别器")


