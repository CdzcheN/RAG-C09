#!/usr/bin/env python3
"""稠密检索（可选支线，用于与 BM25 对比）。

负责人: A
对应文档: docs/项目启动与实施指南.md §1.3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def build_dense(store: Sequence[Mapping[str, Any]], model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """用 sentence-transformers 编码语料构建向量索引。

    状态: 待实现 —— 可选任务：faiss 不可用时改用内存点积检索
    """
    raise NotImplementedError("可选任务：faiss 不可用时改用内存点积检索")


