#!/usr/bin/env python3
"""槽位识别与矛盾值生成（算子 B 的辅助模块）。

负责人: A
对应文档: docs/数据构造规范.md §2.2
说明: 由 `challenge_builder.apply_evidence_conflict` 调用。所有提取器遵循**保守原则**：
    无法可靠识别槽位时返回 None（宁可少构造，也不生成不可判定的样本，规范 §3、§6）。
"""
from __future__ import annotations

import random
import re
from typing import Any, Callable, Mapping, Sequence

#: 年份（时间敏感事实的判定依据之一）
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
#: 数量（排除年份）
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
#: 人名/实体（连续首字母大写词）
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")

SlotHit = tuple[str, str]
SlotFinder = Callable[[str, str, Sequence[Mapping[str, Any]], random.Random], "SlotHit | None"]


def find_date(sentence: str, answer: str, rng: random.Random) -> SlotHit | None:
    """年份槽：优先用答案里的年份，替换为同期不同年份（表面可信）。"""
    for text in (answer, sentence):
        match = YEAR_RE.search(text or "")
        if match:
            year = int(match.group(1))
            new_year = year + rng.choice([-3, -2, -1, 1, 2, 3])
            if 1000 <= new_year <= 2099 and new_year != year:
                return match.group(1), str(new_year)
    return None


def find_number(sentence: str, answer: str, rng: random.Random) -> SlotHit | None:
    """数量槽：排除年份后取第一个数字，做确定性改写（+1 或 ×2）。"""
    for text in (answer, sentence):
        for match in NUMBER_RE.finditer(text or ""):
            raw = match.group(0)
            if YEAR_RE.fullmatch(raw):
                continue
            value = float(raw)
            new_value = value + 1 if rng.random() < 0.5 else value * 2
            rendered = str(int(new_value)) if new_value.is_integer() else f"{new_value:g}"
            if rendered != raw:
                return raw, rendered
    return None


def find_person(sentence: str, answer: str, context: Sequence[Mapping[str, Any]],
                rng: random.Random) -> SlotHit | None:
    """人名槽（保守）：仅当答案本身像人名且字面出现在支撑句中，替换为语料中另一个实体。"""
    if not answer or not NAME_RE.fullmatch(answer.strip()) or answer not in sentence:
        return None
    others: set[str] = set()
    for para in context:
        for sent in para.get("sentences") or []:
            sent = str(sent)
            for match in NAME_RE.finditer(sent):
                candidate = match.group(0)
                if candidate != answer and not sent.startswith(candidate):
                    others.add(candidate)
    if not others:
        return None
    return answer, rng.choice(sorted(others))


def find_place(sentence: str, answer: str, context: Sequence[Mapping[str, Any]],
               rng: random.Random) -> SlotHit | None:
    """地点槽（保守）：仅当答案字面出现在支撑句里，替换为语料中的另一个段落主体。"""
    if not answer or answer not in sentence:
        return None
    if YEAR_RE.search(answer) or NUMBER_RE.fullmatch(answer.strip()):
        return None
    titles = sorted({str(p.get("title", "")) for p in context
                     if p.get("title") and str(p.get("title")) not in sentence})
    if not titles:
        return None
    return answer, rng.choice(titles)


#: 槽位类型 → 提取器（接受统一的 4 参数签名）
SLOT_FINDERS: dict[str, SlotFinder] = {
    "date": lambda sent, ans, ctx, rng: find_date(sent, ans, rng),
    "number": lambda sent, ans, ctx, rng: find_number(sent, ans, rng),
    "person": find_person,
    "place": find_place,
}
