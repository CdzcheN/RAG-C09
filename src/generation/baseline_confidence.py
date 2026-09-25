#!/usr/bin/env python3
"""自评置信度基线（对照 B1）。

负责人: B
对应文档: docs/实验与评估规范.md §2
状态: 已实现（T2.1）

方法来源: 模型自评/内部校准一类工作（Kadavath et al. 2022 的 P(True) 提问式自评，见
`refs/文献综述.md` §5.1）。**方法复现的边界**：这些文献报告的是校准误差与选择性预测类指标，
并未给出与本任务可比的 AUC，因此本实现只做"按原方法实现自评置信度并作为判别分数"，报告中
不得声称复现了某个具体 AUC 数值。

用法: `1 − baseline_confidence` 作为幻觉分数（对照 B1），与一致性特征组严格分开评估。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)

#: P(True) 式自评提示（要求模型只回答 yes/no）
SELF_EVAL_TEMPLATE = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer correct? Answer yes or no.\n"
    "Answer:"
)

YES_WORDS = ("yes", "Yes", "YES")
NO_WORDS = ("no", "No", "NO")


def _candidate_ids(tokenizer: Any, words: Sequence[str]) -> list[int]:
    """把若干写法映射到候选 token id（优先单 token 写法）。"""
    ids: set[int] = set()
    for word in words:
        for variant in (word, f" {word}"):
            try:
                encoded = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:  # noqa: BLE001 - 不同分词器的签名差异不应中断实验
                continue
            if encoded:
                ids.add(int(encoded[0]))
                ids.add(int(encoded[-1]))
    return sorted(ids)


def _normalized_yes_prob(logits: Any, yes_ids: Sequence[int], no_ids: Sequence[int]) -> float:
    """在 {yes, no} 两组 token 上归一化，返回 P(yes)（数值上用 max 平移防溢出）。"""
    yes_logit = max(float(logits[i]) for i in yes_ids)
    no_logit = max(float(logits[i]) for i in no_ids)
    top = max(yes_logit, no_logit)
    yes_exp = math.exp(yes_logit - top)
    no_exp = math.exp(no_logit - top)
    total = yes_exp + no_exp
    return yes_exp / total if total > 0 else float("nan")


def self_confidence(model: Any, tokenizer: Any, question: str, answer: str) -> float:
    """让模型自评该答案正确的概率（P(True) 式），作为幻觉判别的对照分数。

    返回 ∈[0,1]；空答案返回 0.0；无法计算（缺 yes/no 词表、前向失败）时返回 NaN 并记日志，
    由下游在报告中标注（契约允许概率列取 NaN）。
    """
    if not (answer or "").strip():
        return 0.0
    try:
        from ..generation.decode import postprocess  # 局部导入避免不必要的依赖链

        prompt = SELF_EVAL_TEMPLATE.format(question=question, answer=postprocess(answer))
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        import torch

        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, :]

        yes_ids = _candidate_ids(tokenizer, YES_WORDS)
        no_ids = _candidate_ids(tokenizer, NO_WORDS)
        if not yes_ids or not no_ids:
            LOG.warning("分词器缺少 yes/no token，自评置信度记为 NaN")
            return float("nan")
        return _normalized_yes_prob(logits, yes_ids, no_ids)
    except Exception as exc:  # noqa: BLE001 - 自评失败不应中断整条管道
        LOG.warning("自评置信度计算失败（%r），记为 NaN", exc)
        return float("nan")
