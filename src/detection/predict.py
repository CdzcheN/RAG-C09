#!/usr/bin/env python3
"""推理与产物落盘。

负责人: C
对应文档: docs/接口契约.md §2.4
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def predict(exp_id: str, model: Any, features: Sequence[FeatureRow], feature_cols: Sequence[str]) -> list[dict[str, Any]]:
    """输出幻觉概率并组装 MetricRecord 所需字段。

    状态: 待实现 —— T4.1：必须同时记录 config_hash 与 env_report 路径
    """
    raise NotImplementedError("T4.1：必须同时记录 config_hash 与 env_report 路径")


