#!/usr/bin/env python3
"""端到端管道：问题 → 检索 → 提示 → 生成 → 结构化记录。

负责人: B
对应文档: docs/项目启动与实施指南.md §2.2
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def run_pipeline(samples: Sequence[RAGSample], cfg: Mapping[str, Any], seed: int, limit: int | None = None) -> list[Prediction]:
    """对每条样本跑完整管道，返回 Prediction 列表（含检索证据、提示、延迟）。

    状态: 待实现 —— T1.3：门禁 G1——同种子连续两次运行输出逐字节一致
    """
    raise NotImplementedError("T1.3：门禁 G1——同种子连续两次运行输出逐字节一致")


