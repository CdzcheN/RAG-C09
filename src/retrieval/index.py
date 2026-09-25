#!/usr/bin/env python3
"""语料切块与检索索引语料库。

负责人: A
对应文档: docs/项目启动与实施指南.md §2.1
状态: 已实现（T1.2）

口径: 检索语料库是**逐样本**的——每条样本只在自己的 `passages_for_retrieval` 上检索（把别的
样本的段落当作候选没有语义意义）。调用方按样本建库、建索引、检索，见 `bm25.py` 的用法示例。
切块粒度影响 Recall@k 量级，默认句级（`sentence`）并在报告中说明（数据构造规范 §5）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.schema import RAGSample
from ..common.text import split_sentences

GRANULARITIES = ("sentence", "paragraph")


def build_passage_store(samples: Sequence[RAGSample], granularity: str = "sentence") -> list[dict[str, Any]]:
    """把证据上下文切成段/句级 passage，作为检索语料库。

    返回的每条 passage 含 `passage_id`（`<sample_id>#<段落号>#<句号>`）、`sample_id`、`title`、
    `text`。`granularity="paragraph"` 时以整段为单位（压缩上下文长度，但 Recall 粒度更粗）。
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"未知切块粒度：{granularity!r}（可选 {GRANULARITIES}）")

    store: list[dict[str, Any]] = []
    for sample in samples:
        passages = list(sample.passages_for_retrieval or [])
        if not passages:  # 退化兜底：没有候选段落时用 gold 证据句充当语料
            passages = [{"title": "", "text": text} for text in (sample.gold_context or [])]
        for para_idx, para in enumerate(passages):
            title = str(para.get("title", ""))
            text = str(para.get("text", ""))
            if granularity == "sentence":
                units = split_sentences(text) or ([text] if text.strip() else [])
            else:
                units = [text] if text.strip() else []
            for sent_idx, unit in enumerate(units):
                store.append({
                    "passage_id": f"{sample.sample_id}#{para_idx}#{sent_idx}",
                    "sample_id": sample.sample_id,
                    "title": title,
                    "text": unit,
                    "granularity": granularity,
                })
    return store


def store_for_sample(store: Sequence[Mapping[str, Any]], sample_id: str) -> list[dict[str, Any]]:
    """取出某条样本的语料子集（逐样本检索的基本操作）。"""
    return [dict(p) for p in store if p.get("sample_id") == sample_id]
