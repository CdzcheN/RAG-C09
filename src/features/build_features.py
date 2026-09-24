#!/usr/bin/env python3
"""特征表构建与落盘（features.parquet）。

负责人: B
对应文档: docs/接口契约.md §2.3、§6
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def build_feature_table(predictions: Sequence[Prediction], samples: Sequence[RAGSample], nli: Any, cfg: Mapping[str, Any]) -> list[FeatureRow]:
    """串联三组特征，产出契约固定列的 FeatureRow 列表。

    状态: 待实现 —— T3.3：列名必须与 schema.FEATURE_COLUMNS 完全一致，落盘后过 validate
    """
    raise NotImplementedError("T3.3：列名必须与 schema.FEATURE_COLUMNS 完全一致，落盘后过 validate")


