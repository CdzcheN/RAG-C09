#!/usr/bin/env python3
"""NLI 蕴含特征（答案 ← 证据）。

负责人: B
对应文档: docs/实验与评估规范.md §1
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def load_nli(model_id: str = "cross-encoder/nli-deberta-v3-xsmall", device: str = "cuda") -> Any:
    """装载交叉编码器 NLI 模型。

    状态: 待实现 —— T3.2：与生成模型分时复用显存
    """
    raise NotImplementedError("T3.2：与生成模型分时复用显存")


def entailment_scores(answer: str, evidence: Sequence[str], model: Any, tokenizer: Any) -> dict[str, float]:
    """计算蕴含/矛盾概率的最大值与均值。

    状态: 待实现 —— T3.2：产出 entailment_max / entailment_mean / contradiction_max
    """
    raise NotImplementedError("T3.2：产出 entailment_max / entailment_mean / contradiction_max")


