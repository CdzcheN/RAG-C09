#!/usr/bin/env python3
"""检索质量评估：Recall@k。

负责人: A
对应文档: docs/数据构造规范.md §5
状态: 已实现（T1.2）

口径: 逐查询计算 `|gold ∩ 检索前 k 条| / |gold|`，再对所有**有 gold 标注**的查询取平均；无 gold
标注的查询不计入（分母只算有效查询）。证据匹配在句级粒度上按规范化文本相等判定。
"""
from __future__ import annotations

from typing import Sequence


def recall_at_k(retrieved: Sequence[Sequence[str]], gold: Sequence[set[str]], k: int) -> float:
    """计算 Recall@k（gold 证据是否出现在前 k 条）。

    `retrieved[i]` 为第 i 个查询的检索结果文本（已规范化或原文），`gold[i]` 为该查询的 gold
    证据文本集合。两序列长度必须一致；无有效查询时返回 NaN（报告中需标注该情况）。
    """
    if len(retrieved) != len(gold):
        raise ValueError("retrieved 与 gold 长度不一致")
    if k <= 0:
        raise ValueError("k 必须为正整数")

    scores: list[float] = []
    for hits, gold_set in zip(retrieved, gold):
        if not gold_set:
            continue
        top_k = set(str(item) for item in list(hits)[:k])
        hits_in_top = len(top_k & {str(g) for g in gold_set})
        scores.append(hits_in_top / len(gold_set))
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def recall_at_k_by_cutoffs(retrieved: Sequence[Sequence[str]], gold: Sequence[set[str]],
                           cutoffs: Sequence[int]) -> dict[str, float]:
    """一次给出多个 k 的 Recall@k（报告用，例如 k=1/3/5）。"""
    return {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in cutoffs}
