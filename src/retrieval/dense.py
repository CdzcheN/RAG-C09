#!/usr/bin/env python3
"""稠密检索（可选支线，用于与 BM25 对比）。

负责人: A
对应文档: docs/项目启动与实施指南.md §1.3
状态: 已实现（可选任务）

实现说明: 用 `sentence-transformers` 编码语料与问题，**内存点积检索**（不依赖 faiss；
requirements.txt 中 faiss-cpu 亦为可选）。默认关闭（`configs/models.yaml dense_retriever.enabled`）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.schema import RetrievedPassage


class DenseIndex:
    """稠密索引：句向量 + 语料（store 与向量必须成对使用）。"""

    def __init__(self, store: Sequence[Mapping[str, Any]], embeddings: Any, model: Any) -> None:
        self.store = list(store)
        self.embeddings = embeddings
        self.model = model

    def __len__(self) -> int:
        return len(self.store)


def build_dense(store: Sequence[Mapping[str, Any]],
                model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> DenseIndex:
    """用 sentence-transformers 编码语料构建向量索引（归一化后点积等价余弦相似度）。

    空语料时返回空索引，`search_dense` 直接返回空列表。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "稠密检索需要 sentence-transformers，请先执行 python -m pip install -r requirements.txt"
        ) from exc

    passages = [dict(p) for p in store]
    model = SentenceTransformer(model_id)
    if not passages:
        return DenseIndex(passages, None, model)
    texts = [str(p.get("text", "")) for p in passages]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)
    return DenseIndex(passages, embeddings, model)


def search_dense(query: str, index: DenseIndex, store: Sequence[Mapping[str, Any]] | None = None,
                 top_k: int = 5) -> list[RetrievedPassage]:
    """稠密检索 top-k（返回契约中的 `RetrievedPassage`，score = 余弦相似度）。"""
    if index is None or index.embeddings is None:
        return []
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("稠密检索需要 numpy（见 requirements.txt）") from exc

    passages = list(store) if store is not None else index.store
    if not passages:
        return []
    query_vec = index.model.encode([query], normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True)[0]
    scores = np.asarray(index.embeddings) @ np.asarray(query_vec)
    order = sorted(range(len(passages)),
                   key=lambda i: (-float(scores[i]), str(passages[i].get("passage_id", ""))))

    hits: list[RetrievedPassage] = []
    for rank, position in enumerate(order[:max(0, top_k)], start=1):
        passage = passages[position]
        hits.append(RetrievedPassage(rank=rank, title=str(passage.get("title", "")),
                                     text=str(passage.get("text", "")),
                                     score=float(scores[position])))
    return hits
