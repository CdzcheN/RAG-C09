#!/usr/bin/env python3
"""证据间一致性：矛盾句对计数与检索分数。

负责人: B
对应文档: docs/接口契约.md §2.3
状态: 已实现（T3.3）

口径: `evidence` 元素可以是 `RetrievedPassage`（含检索分数）或纯文本。
- `conflict_count` = 证据句对中"前提句 → 假设句"被判为矛盾（概率 > 阈值）的对数；
- `retrieval_top_score` = 检索结果中的最高分（无分数信息时为 NaN）。
句对级 NLI 的复杂度是 O(n²)，n ≤ `top_k`（默认 5）→ 最多 10 次前向，可接受。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import RetrievedPassage
from .entailment import label_indices

LOG = get_logger(__name__)

#: 矛盾判定阈值（证据句对之间；与判别阈值 0.5 无关，报告需说明该取值）
CONTRADICTION_THRESHOLD = 0.5


def _texts_and_scores(evidence: Iterable[Any]) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    for item in evidence:
        if isinstance(item, RetrievedPassage):
            texts.append(str(item.text))
            scores.append(float(item.score))
        elif isinstance(item, dict):
            texts.append(str(item.get("text", "")))
            if "score" in item:
                scores.append(float(item["score"]))
        else:
            texts.append(str(item))
    return texts, scores


def conflict_features(evidence: Sequence[Any], nli: Any,
                      threshold: float = CONTRADICTION_THRESHOLD) -> dict[str, float]:
    """统计证据句对之间的矛盾计数，并回收检索最高分（契约列 `conflict_count` / `retrieval_top_score`）。

    `nli` 为 `src.features.entailment.NLIModel`（支持解包）；为 None 时 `conflict_count` 记 NaN。
    """
    nan = float("nan")
    texts, scores = _texts_and_scores(evidence)
    top_score = max(scores) if scores else nan

    clean = [t for t in texts if t.strip()]
    if nli is None or len(clean) < 2:
        return {"conflict_count": 0.0 if nli is not None else nan, "retrieval_top_score": top_score}

    import torch

    model, tokenizer = nli
    _, contradiction_idx = label_indices(model)
    pairs = [[clean[i], clean[j]] for i in range(len(clean)) for j in range(i + 1, len(clean))]
    inputs = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits.float(), dim=-1).cpu()

    count = sum(1 for row in probs if float(row[contradiction_idx]) > threshold)
    return {"conflict_count": float(count), "retrieval_top_score": top_score}
