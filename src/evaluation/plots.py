#!/usr/bin/env python3
"""图表自绘（禁止截图，要求.md 2.3 条第 6 款）。

负责人: C
对应文档: docs/实验与评估规范.md §6
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def plot_grouped_bars(summary: Mapping[str, Mapping[str, float]], out: str) -> str:
    """按幻觉类型绘制 AUC / 查准 / 查全分组柱状图（含误差棒与样本量 n）。

    状态: 待实现 —— T6.2：六张必备图之一
    """
    raise NotImplementedError("T6.2：六张必备图之一")


def plot_roc_pr(y_true: Sequence[int], curves: Mapping[str, Sequence[float]], out: str) -> str:
    """绘制 M / B1 / B2 的 ROC 与 PR 曲线。

    状态: 待实现 —— T6.2：中文字体用 Noto Sans CJK（scripts/make_figures.py 已验证可用）
    """
    raise NotImplementedError("T6.2：中文字体用 Noto Sans CJK（scripts/make_figures.py 已验证可用）")


