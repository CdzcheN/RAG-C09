#!/usr/bin/env python3
"""语料切块与检索索引构建。

负责人: A
对应文档: docs/项目启动与实施指南.md §2.1
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def build_passage_store(samples: Sequence[RAGSample], granularity: str = "sentence") -> list[dict[str, Any]]:
    """把证据上下文切成段/句级 passage，作为检索语料库。

    状态: 待实现 —— T1.2：粒度决定 Recall@k 量级，需在报告中说明所选粒度
    """
    raise NotImplementedError("T1.2：粒度决定 Recall@k 量级，需在报告中说明所选粒度")


