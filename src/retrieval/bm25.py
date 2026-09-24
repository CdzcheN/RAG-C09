#!/usr/bin/env python3
"""BM25 稀疏检索（主管线）。

负责人: A
对应文档: docs/项目启动与实施指南.md §2.1
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def build_bm25(store: Sequence[Mapping[str, Any]]) -> Any:
    """用 rank_bm25.BM25Okapi 建索引（分词用自带正则）。

    状态: 待实现 —— T1.2：避免 nltk 首次运行联网下载 punkt
    """
    raise NotImplementedError("T1.2：避免 nltk 首次运行联网下载 punkt")


def search(query: str, index: Any, store: Sequence[Mapping[str, Any]], top_k: int = 5) -> list[RetrievedPassage]:
    """检索 top-k 证据，返回契约中的 RetrievedPassage 列表。

    状态: 待实现 —— T1.2：同一查询必须得到同一结果（可复现）
    """
    raise NotImplementedError("T1.2：同一查询必须得到同一结果（可复现）")


