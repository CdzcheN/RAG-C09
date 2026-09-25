#!/usr/bin/env python3
"""三类退化算子的实现（`challenge_builder` 的算子层，模块划分见规范 §6「单文件 ≤400 行」）。

负责人: A
对应文档: docs/数据构造规范.md §2
说明: 对外接口仍以 `src.datasets.challenge_builder` 为准（那里 re-export 本模块的三个算子），
    本模块只承载算子细节与段落工具。
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from ..common.schema import RAGSample  # noqa: F401  （类型注释用）
from ..common.text import lexical_overlap
from .slots import SLOT_FINDERS, YEAR_RE


# —— 段落工具 ——

def para_text(para: Mapping[str, Any]) -> str:
    """段落 -> 文本（句列表拼接）。"""
    return " ".join(str(s) for s in (para.get("sentences") or [])).strip()


def para_dict(para: Mapping[str, Any]) -> dict[str, str]:
    """统一为契约 `passages_for_retrieval` 的 `{title, text}` 形式。"""
    if "text" in para and "sentences" not in para:
        return {"title": str(para.get("title", "")), "text": str(para.get("text", ""))}
    return {"title": str(para.get("title", "")), "text": para_text(para)}


def middle_index(n: int) -> int:
    """中部位置（Lost in the Middle 的位置敏感性结论，规范 §2.2）。"""
    return max(1, n // 2)


def build_pool(examples: Sequence[Mapping[str, Any]], start: int, size: int,
               max_per_row: int = 2) -> list[dict[str, str]]:
    """构造 sampled 干扰池：取靠后样本的段落，尽量与被选基样本错开。"""
    pool: list[dict[str, str]] = []
    for example in examples[start:start + size]:
        for para in (example.get("context") or [])[:max_per_row]:
            if para_text(para):
                pool.append(para_dict(para))
    return pool


# —— 算子 A：检索失败 ——

def apply_retrieval_failure(example: Mapping[str, Any], params: Mapping[str, Any],
                            seed: int) -> dict[str, Any] | None:
    """算子 A：移除 gold 证据句并替换为同主题干扰段。

    实现: ① `supporting_facts` 定位 gold 句并整体移除；② 干扰段优先取 HotpotQA 自带的
    distractor 文本（`distractor_pool=native`），按与问题的词面重叠降序选取且不低于
    `min_lexical_overlap`；不足时用外部池（`sampled`）补齐并记 `distractor_shortage`。
    无法移除任何 gold 句时返回 None（该样本不适用）。
    """
    n_replace = int(params.get("n_replace", 2))
    pool_kind = str(params.get("distractor_pool", "native"))
    min_overlap = float(params.get("min_lexical_overlap", 0.2))
    rng = random.Random(seed)

    question = str(example.get("question", ""))
    context = list(example.get("context") or [])
    facts = example.get("supporting_facts") or {}
    gold_pairs = {(str(t), int(s)) for t, s in zip(facts.get("title") or [], facts.get("sent_id") or [])}
    gold_titles = {title for title, _ in gold_pairs}

    kept: list[Mapping[str, Any]] = []
    removed: list[str] = []
    for para in context:
        sentences: list[str] = []
        for idx, sentence in enumerate(para.get("sentences") or []):
            if (str(para.get("title", "")), idx) in gold_pairs:
                removed.append(str(sentence))
            else:
                sentences.append(str(sentence))
        if sentences:
            kept.append({"title": para.get("title", ""), "sentences": sentences})
    if not removed:
        return None

    candidates: list[dict[str, str]] = []
    if pool_kind != "sampled":
        candidates = [para_dict(p) for p in context if str(p.get("title", "")) not in gold_titles]
    if len(candidates) < n_replace:
        candidates += [dict(c) for c in (params.get("_pool") or [])]

    scored = [(lexical_overlap(question, c["text"]), c) for c in candidates if c.get("text")]
    scored.sort(key=lambda item: (-item[0], rng.random()))
    chosen = [c for score, c in scored if score >= min_overlap][:n_replace]
    shortage = len(chosen) < n_replace
    if shortage:  # 无法满足重叠下限时放宽，并如实记录（规范 §6 的对策）
        chosen = [c for _, c in scored[:n_replace]]

    passages = [para_dict(p) for p in kept] + chosen
    return {
        "challenge_type": "retrieval_failure",
        "question": question,
        "gold_answer": str(example.get("answer", "")),
        "gold_context": tuple(str(s) for s in (example.get("gold_context") or [])),
        "passages_for_retrieval": tuple(passages),
        "injected_values": (),
        "is_hallucination": 1,
        "construct_params": {
            "operator": "retrieval_failure", "seed": seed, "n_replace": n_replace,
            "distractor_pool": pool_kind, "min_lexical_overlap": min_overlap,
            "n_removed_sentences": len(removed), "removed_from_corpus": True,
            "n_injected_passages": len(chosen), "distractor_shortage": shortage,
            "distractor_overlaps": sorted(round(lexical_overlap(question, c["text"]), 4) for c in chosen),
        },
    }


# —— 算子 B：证据冲突 ——

def apply_evidence_conflict(example: Mapping[str, Any], params: Mapping[str, Any],
                            seed: int) -> dict[str, Any] | None:
    """算子 B：在上下文中部注入与真值矛盾的槽位句。

    实现: ① 在支撑句中按 `slot_type` 顺序识别可替换槽位（人名/日期/数量/地点，见 `slots.py`），
    用同类实体替换生成矛盾句；② 注入位置 `inject_position`（默认 `middle`）；③ 提取不到任何
    槽位时返回 None，不做猜测。
    """
    slot_types = list(params.get("slot_type") or ["person", "date", "number", "place"])
    n_inject = int(params.get("n_inject", 1))
    position = str(params.get("inject_position", "middle"))
    rng = random.Random(seed)

    context = list(example.get("context") or [])
    answer = str(example.get("answer", ""))
    gold_sentences = [str(s) for s in (example.get("gold_context") or [])]
    if not context or not gold_sentences:
        return None
    base_sentence = gold_sentences[0]

    injections: list[dict[str, str]] = []
    used_slots: set[str] = set()
    for slot in slot_types:
        if len(injections) >= n_inject:
            break
        if slot in used_slots or slot not in SLOT_FINDERS:
            continue
        found = SLOT_FINDERS[slot](base_sentence, answer, context, rng)
        if not found:
            continue
        original, replacement = found
        injections.append({"slot_type": slot, "original": original, "injected": replacement,
                           "sentence": base_sentence.replace(original, replacement, 1)})
        used_slots.add(slot)
    if not injections:
        return None

    patched: list[dict[str, str]] = []
    for para in context:
        sentences = [str(s) for s in para.get("sentences") or []]
        if base_sentence in sentences:
            at = sentences.index(base_sentence)
            insert_at = (min(at + middle_index(len(sentences)), len(sentences))
                         if position == "middle" else len(sentences))
            for injection in reversed(injections):
                sentences.insert(insert_at, injection["sentence"])
        patched.append({"title": str(para.get("title", "")), "text": " ".join(sentences)})

    return {
        "challenge_type": "evidence_conflict",
        "question": str(example.get("question", "")),
        "gold_answer": answer,
        "gold_context": tuple(gold_sentences),
        "passages_for_retrieval": tuple(patched),
        "injected_values": tuple(inj["injected"] for inj in injections),
        "is_hallucination": 1,
        "construct_params": {
            "operator": "evidence_conflict", "seed": seed, "n_inject": n_inject,
            "inject_position": position,
            "injections": [dict(inj) for inj in injections],
        },
    }


# —— 算子 C：答案过时 ——

def apply_outdated(example: Mapping[str, Any], params: Mapping[str, Any],
                   seed: int) -> dict[str, Any] | None:
    """算子 C：把时间敏感事实改写为过期版本。

    实现: 在支撑句或答案中定位年份，按 `time_gap_years` 回退改写证据（证据自洽、但相对当前
    真值已过时）；找不到年份的样本返回 None，由调用方计数并按指南 §6 降级报告。
    """
    gap = int(params.get("time_gap_years", 2))
    context = list(example.get("context") or [])
    answer = str(example.get("answer", ""))
    gold_sentences = [str(s) for s in (example.get("gold_context") or [])]
    if not context or not gold_sentences:
        return None
    base_sentence = gold_sentences[0]
    match = YEAR_RE.search(base_sentence) or YEAR_RE.search(answer)
    if not match:
        return None
    current_year = int(match.group(1))
    outdated_year = current_year - gap
    if outdated_year < 1000:
        return None
    outdated_sentence = base_sentence.replace(match.group(1), str(outdated_year), 1)

    patched: list[dict[str, str]] = []
    replaced = False
    for para in context:
        sentences = [str(s) for s in para.get("sentences") or []]
        if not replaced and base_sentence in sentences:
            sentences[sentences.index(base_sentence)] = outdated_sentence
            replaced = True
        patched.append({"title": str(para.get("title", "")), "text": " ".join(sentences)})

    outdated_answer = (answer.replace(match.group(1), str(outdated_year), 1)
                       if match.group(1) in answer else str(outdated_year))
    return {
        "challenge_type": "outdated",
        "question": str(example.get("question", "")),
        "gold_answer": answer,
        "gold_context": tuple(gold_sentences),
        "passages_for_retrieval": tuple(patched),
        "injected_values": (outdated_answer,),
        "is_hallucination": 1,
        "construct_params": {
            "operator": "outdated", "seed": seed, "time_gap_years": gap,
            "gold_version": match.group(1), "outdated_version": str(outdated_year),
            "patched_sentence": outdated_sentence,
        },
    }


def make_control(example: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """对照版：同一基样本、证据完好（无任何退化），与污染版成对（规范 §1）。"""
    context = list(example.get("context") or [])
    gold_context = tuple(str(s) for s in (example.get("gold_context") or []))
    if not gold_context and context:
        gold_context = tuple(str(s) for s in (context[0].get("sentences") or [])[:1])
    return {
        "challenge_type": "none",
        "question": str(example.get("question", "")),
        "gold_answer": str(example.get("answer", "")),
        "gold_context": gold_context,
        "passages_for_retrieval": tuple(para_dict(p) for p in context),
        "injected_values": (),
        "is_hallucination": 0,
        "construct_params": {"operator": "none", "seed": seed, "is_control": True},
    }
