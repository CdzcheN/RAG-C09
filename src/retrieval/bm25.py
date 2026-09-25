#!/usr/bin/env python3
"""BM25 稀疏检索（主管线）。

负责人: A
对应文档: docs/项目启动与实施指南.md §2.1
状态: 已实现（T1.2）

用法（逐样本检索）:

    store = build_passage_store([sample])
    index = build_bm25(store)
    hits = search(sample.question, index, store, top_k=5)

可复现: BM25Okapi 本身确定，同分时按 `passage_id` 稳定排序，因此同一查询必得同一结果（门禁 G1）。
分词用自带正则（`src/common/text.py`），避免 nltk 首次运行联网下载 punkt。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.schema import RetrievedPassage
from ..common.text import tokenize


def _bm25_class():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "BM25 检索需要 rank_bm25，请先执行 python -m pip install -r requirements.txt"
        ) from exc
    return BM25Okapi


class BM25Index:
    """BM25 索引与语料的绑定（store 与索引必须成对使用）。"""

    def __init__(self, store: Sequence[Mapping[str, Any]], index: Any) -> None:
        self.store = list(store)
        self.index = index

    def __len__(self) -> int:
        return len(self.store)


def build_bm25(store: Sequence[Mapping[str, Any]]) -> BM25Index:
    """用 `rank_bm25.BM25Okapi` 建索引（分词用自带正则）。

    空语料时返回仅含空索引的对象，`search` 会直接返回空列表（而不是抛错）。
    """
    passages = [dict(p) for p in store]
    if not passages:
        return BM25Index(passages, None)
    corpus = [tokenize(str(p.get("text", ""))) for p in passages]
    BM25Okapi = _bm25_class()
    return BM25Index(passages, BM25Okapi(corpus))


def search(query: str, index: BM25Index, store: Sequence[Mapping[str, Any]] | None = None,
           top_k: int = 5) -> list[RetrievedPassage]:
    """检索 top-k 证据，返回契约中的 `RetrievedPassage` 列表（rank 从 1 开始）。

    `store` 可省略（默认用 `index.store`）；给出时必须与建索引时的语料一致。
    """
    if index is None or index.index is None:
        return []
    passages = list(store) if store is not None else index.store
    if not passages:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.index.get_scores(tokens)
    order = sorted(range(len(passages)), key=lambda i: (-float(scores[i]), str(passages[i].get("passage_id", ""))))

    hits: list[RetrievedPassage] = []
    for rank, position in enumerate(order[:max(0, top_k)], start=1):
        passage = passages[position]
        hits.append(RetrievedPassage(rank=rank, title=str(passage.get("title", "")),
                                     text=str(passage.get("text", "")),
                                     score=float(scores[position])))
    return hits
