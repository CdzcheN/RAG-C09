#!/usr/bin/env python3
"""生成模型装载与显存管理。

负责人: B
对应文档: docs/项目启动与实施指南.md §1.3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def load_generator(model_id: str, dtype: str = "float16", device: str = "cuda") -> Any:
    """装载因果语言模型与分词器（fp16、低显存策略）。

    状态: 待实现 —— T1.3：可用显存约 3.68 GB，batch 固定为 1
    """
    raise NotImplementedError("T1.3：可用显存约 3.68 GB，batch 固定为 1")


def vram_report() -> dict[str, float]:
    """报告已用/可用显存，便于提前判断 OOM 风险。

    状态: 待实现 —— T1.3
    """
    raise NotImplementedError("T1.3")


