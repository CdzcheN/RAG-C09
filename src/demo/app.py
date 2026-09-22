#!/usr/bin/env python3
"""演示入口（复用实验代码，不另写一套逻辑）。

负责人: B
对应文档: docs/编码与协作规范.md §2.4
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def answer_with_evidence(question: str, cfg_path: str = "configs/default.yaml") -> dict[str, Any]:
    """跑一次完整管道并返回答案、证据与幻觉分数，供命令行/界面展示。

    状态: 待实现 —— T6.3：必须与 src/generation/pipeline.py 共用实现
    """
    raise NotImplementedError("T6.3：必须与 src/generation/pipeline.py 共用实现")


