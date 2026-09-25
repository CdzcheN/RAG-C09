#!/usr/bin/env python3
"""轻量文本工具：分词、规范化、句切分（纯标准库，无第三方依赖）。

负责人: A（公共层，主责接口）
说明: 分词口径与 `src/features/overlap.py` 保持一致（小写化 + 字母数字切分），
    避免 nltk 首次运行联网下载 punkt（见 docs/项目启动与实施指南.md §1.2）。
    句切分按句末标点近似切分：缩写（如 "U.S. Army"）可能误切，对句级检索与支撑句定位
    是可接受的近似，原型阶段需人工确认（见 docs/数据构造规范.md §6）。
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")


def tokenize(text: str) -> list[str]:
    """小写化后按字母数字切分。"""
    return TOKEN_RE.findall((text or "").lower())


def normalize(text: str) -> str:
    """规范化：小写 + 去标点 + 规整空白。"""
    return " ".join(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """把段落切成句子列表；空文本返回空列表。"""
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def contains(haystack: str, needle: str) -> bool:
    """规范化后的按词边界包含判定（避免 "yes" 命中 "yesterday"）。

    非 ASCII 关键词（如中文）在规范化时会被 `TOKEN_RE` 丢弃，因此对这类关键词退化为原文小写
    子串匹配——否则中文拒答模板（"无法回答"）与中文答案将永远匹配不上。
    """
    h, n = normalize(haystack), normalize(needle)
    if h and n and (h == n or f" {n} " in f" {h} "):
        return True
    if not haystack or not needle:
        return False
    if any(ord(ch) > 127 for ch in needle):
        return needle.strip().lower() in haystack.lower()
    return False


def lexical_overlap(query: str, text: str) -> float:
    """query 的词被 text 覆盖的比例（∈[0,1]）；query 无词时返回 0.0。"""
    qtokens = set(tokenize(query))
    if not qtokens:
        return 0.0
    return len(qtokens & set(tokenize(text))) / len(qtokens)
