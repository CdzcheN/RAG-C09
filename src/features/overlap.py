#!/usr/bin/env python3
"""词面重叠特征：精确匹配、token-F1、证据覆盖率。

负责人: B ｜ 对应文档: docs/实验与评估规范.md §1、docs/接口契约.md §2.3
说明: 纯函数、无重依赖，供 build_features 与单元测试直接调用。
    分词采用自带正则，避免 nltk 首次运行需联网下载 punkt（见 docs/项目启动与实施指南.md §1.2）。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """小写化后按字母数字切分。"""
    return TOKEN_RE.findall((text or "").lower())


def normalize(text: str) -> str:
    """规范化：小写 + 去标点 + 规整空白。"""
    return " ".join(tokenize(text))


def exact_match(prediction: str, gold: str) -> float:
    """规范化后完全一致返回 1.0，否则 0.0（契约列 overlap_em）。"""
    return 1.0 if normalize(prediction) == normalize(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD 式 token 级 F1（多重集交集，契约列 overlap_f1）。"""
    pred_tokens, gold_tokens = tokenize(prediction), tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return float(normalize(prediction) == normalize(gold))
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_coverage(answer: str, evidence: Iterable[str]) -> float:
    """答案中内容词被证据覆盖的比例（契约列 citation_cov）；答案为空时返回 0.0。

    说明: 当前按全部 token 计算、**不做停用词过滤**，因此对 "in / the / of" 这类
    高频词的覆盖会偏乐观；若消融显示该特征区分度不足，可在此加入停用词表或改用
    内容词（名词/数字）子集，并同步更新 tests/test_overlap.py 的期望值。
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for sentence in evidence:
        evidence_tokens.update(tokenize(sentence))
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def overlap_features(answer: str, gold: str, evidence: Sequence[str]) -> dict[str, float]:
    """打包为契约中的三个重叠特征列。"""
    return {
        "overlap_em": exact_match(answer, gold),
        "overlap_f1": token_f1(answer, gold),
        "citation_cov": citation_coverage(answer, evidence),
    }
