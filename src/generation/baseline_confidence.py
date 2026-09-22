#!/usr/bin/env python3
"""自评置信度基线（对照 B1）。

负责人: B
对应文档: docs/实验与评估规范.md §2
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def self_confidence(model: Any, tokenizer: Any, question: str, answer: str) -> float:
    """让模型自评该答案正确的概率（P(True) 式），作为幻觉判别的对照分数。

    状态: 待实现 —— T2.1：方法复现（自评/校准一类工作）；报告中须写明无法与原文 AUC 直接比较
    """
    raise NotImplementedError("T2.1：方法复现（自评/校准一类工作）；报告中须写明无法与原文 AUC 直接比较")


