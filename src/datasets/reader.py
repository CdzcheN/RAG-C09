#!/usr/bin/env python3
"""数据读取：按数据集标识在线加载并做字段规范化。

负责人: A
对应文档: docs/项目启动与实施指南.md §1.4、docs/数据构造规范.md §1
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def load_dataset_split(dataset_id: str, config: str | None = None, split: str = "validation", revision: str | None = None) -> Any:
    """加载指定数据集的某个划分；revision 固定后写入产物元信息（契约 §4）。

    状态: 待实现 —— T1.1：用 datasets.load_dataset 实现，禁止手动下载
    """
    raise NotImplementedError("T1.1：用 datasets.load_dataset 实现，禁止手动下载")


def normalize_hotpotqa(example: Mapping[str, Any]) -> dict[str, Any]:
    """把 HotpotQA 样本规范化为 question / answer / context / supporting_facts 结构。

    状态: 待实现 —— T1.1：统一字段名，供 challenge_builder 消费
    """
    raise NotImplementedError("T1.1：统一字段名，供 challenge_builder 消费")


def normalize_squad(example: Mapping[str, Any]) -> dict[str, Any]:
    """把 SQuAD v2 样本规范化为同一结构（保留不可回答标记）。

    状态: 待实现 —— T1.1：不可回答问题用于“证据缺失”类对照
    """
    raise NotImplementedError("T1.1：不可回答问题用于“证据缺失”类对照")


