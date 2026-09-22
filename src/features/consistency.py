#!/usr/bin/env python3
"""证据间一致性：矛盾句对计数与检索分数。

负责人: B
对应文档: docs/接口契约.md §2.3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def conflict_features(evidence: Sequence[str], nli: Any) -> dict[str, float]:
    """统计证据句对之间的矛盾计数，并回收检索最高分。

    状态: 待实现 —— T3.3：产出 conflict_count 与 retrieval_top_score
    """
    raise NotImplementedError("T3.3：产出 conflict_count 与 retrieval_top_score")


