#!/usr/bin/env python3
"""抽样与划分：常规集抽样、按基样本切分 dev-fit / dev-test。

负责人: A
对应文档: docs/数据构造规范.md §1
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def sample_normal_set(cfg: Mapping[str, Any], seed: int) -> list[RAGSample]:
    """从 SQuAD v2 dev 与 HotpotQA validation 各抽样 N 条构成常规集。

    状态: 待实现 —— T1.1：抽样种子显式传入并写入产物元信息
    """
    raise NotImplementedError("T1.1：抽样种子显式传入并写入产物元信息")


def split_by_base_sample(samples: Sequence[RAGSample], ratio: Mapping[str, float], seed: int) -> tuple[list[RAGSample], list[RAGSample]]:
    """按基样本切分，污染版与对照版必须同侧（避免数据泄漏）。

    状态: 待实现 —— T3.1：ratio 见 configs/default.yaml sampling.split_ratio
    """
    raise NotImplementedError("T3.1：ratio 见 configs/default.yaml sampling.split_ratio")


