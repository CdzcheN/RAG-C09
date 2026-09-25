#!/usr/bin/env python3
"""挑战子集构造：三类退化算子与标签规则化推导。

负责人: A
对应文档: docs/数据构造规范.md §2、§3
状态: 已实现（T2.1/T2.2/T3.1）

模块划分: 三个算子与段落工具实现在 `src/datasets/operators.py`，槽位提取在 `src/datasets/slots.py`，
构造日志汇总在 `src/datasets/construct_log.py`（规范 §6「单文件 ≤400 行」）。本模块是三者对外的
统一入口，并在下方 re-export，保证 `from src.datasets.challenge_builder import apply_*` 可用。

口径约定（与契约字段含义对齐，需全组知悉）:

1. `gold_answer` / `gold_context` 记录**真值答案及其支撑证据句**：即使检索失败算子已把支撑句从
   语料中移除（`construct_params.removed_from_corpus=true`）、或证据冲突算子注入了矛盾句
   （`construct_params.injected_values=[...]`），`gold_context` 仍是真值所依据的句子，供标签推导
   与人工核对使用；构造对语料的实际改动记在 `construct_params`。
2. `split="challenge"`；每类 `n_base_per_type` 个基样本各产出**污染版**与**对照版**：
   污染版 `challenge_type=<算子名>`、`construct_params.is_control=false`；对照版
   `challenge_type="none"`（证据完好、无退化）、`construct_params.is_control=true`；两者
   `provenance.row_index` 相同，便于成对比较且 `sample_id` 不冲突（契约 §2.1 命名规则）。
3. `is_hallucination` 在此阶段写**构造先验**（污染版 1 / 对照版 0），并记
   `construct_params.label_source="construct_prior"`；最终标签由 `derive_label(模型答案)` 回填
   （规范 §3 的判定需要模型答案）。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import RAGSample
from ..common.seeding import DATA_SEED_BASE
from ..common.text import contains
from .construct_log import build_construct_log, summarize_challenge_set  # noqa: F401  (re-export)
from .operators import (apply_evidence_conflict, apply_outdated,  # noqa: F401  (re-export)
                        apply_retrieval_failure, build_pool, make_control)

LOG = get_logger(__name__)

#: 拒答模板（规范 §3：命中即视为"未给出具体答案"）
REFUSAL_PATTERNS = (
    "无法回答", "不能回答", "不知道", "不确定", "没有足够信息", "信息不足", "无法确定", "无答案",
    "insufficient information", "insufficient context", "cannot answer", "can not answer",
    "can't answer", "unable to answer", "i don't know", "i do not know", "do not know",
    "not enough information", "not answerable", "no answer", "unknown",
)

#: 三类算子（顺序即候选样本的消费顺序，保证三类基样本互不重叠）
CHALLENGE_OPERATORS = ("retrieval_failure", "evidence_conflict", "outdated")


def mentions(answer: str, value: str) -> bool:
    """答案是否提到某个值（规范化后相等或按词边界包含）。"""
    return contains(answer, value)


def is_refusal(answer: str) -> bool:
    """空答案或命中拒答模板 → 视为"未给出具体答案"。

    ASCII 模板按规范化词边界匹配（避免 "unknown" 命中 "unknowns" 之类的假阳性），中文等非
    ASCII 模板按原文子串匹配（规范化会丢弃非 ASCII 字符，见 `common.text.contains`）。
    """
    text = (answer or "").strip()
    if not text:
        return True
    return any(contains(text, pattern) for pattern in REFUSAL_PATTERNS)


def derive_label(answer: str, gold_answer: str, injected_values: Sequence[str],
                 challenge_type: str, is_control: bool = False) -> int | None:
    """规则化标签推导；无法判定返回 None（记 undetermined 并排除）。

    规则（数据构造规范 §3，标签 = 1 − is_supported）:
      - 检索失败：拒答或空答案 → 0；给出任何具体答案 → 1；
      - 证据冲突：匹配注入的错误值 → 1；匹配 gold → 0；两者都不匹配 → None；
      - 答案过时：匹配过期版本 → 1；匹配当前版本 → 0；否则 None；
      - 对照版（`is_control=True`）与常规集：匹配 gold → 0；否则 → 1。
    禁止引入 NLI 等模型信号（否则标签与特征同源，属循环论证）。
    """
    if challenge_type == "retrieval_failure" and not is_control:
        return 0 if is_refusal(answer) else 1
    if not is_control and challenge_type in ("evidence_conflict", "outdated"):
        if any(mentions(answer, value) for value in injected_values):
            return 1
        if mentions(answer, gold_answer):
            return 0
        return None
    return 0 if mentions(answer, gold_answer) else 1


def _to_sample(example: Mapping[str, Any], result: Mapping[str, Any], *, is_control: bool,
               operator: str, dataset: Mapping[str, Any] | None = None) -> RAGSample:
    """把算子结果组装为契约 RAGSample（命名规则见契约 §2.1）。"""
    challenge_type = str(result["challenge_type"])
    row_index = int(example.get("row_index", -1))
    source = str(example.get("source", "unknown"))
    split_tag = str(example.get("split_tag", "dev"))
    sample_id = f"{source}-{split_tag}-{row_index:05d}-{challenge_type}"

    construct_params = dict(result.get("construct_params") or {})
    construct_params.update({
        "challenge_type": challenge_type,
        "operator": operator,
        "pair_type": "control" if is_control else "polluted",
        "is_control": is_control,
        "injected_values": list(result.get("injected_values") or []),
        "label_source": "construct_prior",
        "label_rule": "最终标签由 derive_label(模型答案) 回填（数据构造规范 §3）",
    })
    provenance: dict[str, Any] = {
        "row_index": row_index,
        "hf_split": example.get("hf_split", "dev"),
        "source_id": example.get("id", ""),
        "is_impossible": bool(example.get("is_impossible", False)),
        "base_sample_key": f"{source}-{split_tag}-{row_index:05d}",
    }
    if dataset:
        provenance["dataset"] = dict(dataset)
    return RAGSample(
        sample_id=sample_id,
        source=source,
        split="challenge",
        challenge_type=challenge_type,
        question=str(result.get("question", "")),
        gold_answer=str(result.get("gold_answer", "")),
        gold_context=tuple(result.get("gold_context") or ()),
        passages_for_retrieval=tuple(result.get("passages_for_retrieval") or ()),
        is_hallucination=int(result.get("is_hallucination", 0)),
        construct_params=construct_params,
        provenance=provenance,
    )


def _operator_fn(challenge_type: str) -> Callable[..., dict[str, Any] | None]:
    return {"retrieval_failure": apply_retrieval_failure,
            "evidence_conflict": apply_evidence_conflict,
            "outdated": apply_outdated}[challenge_type]


def build_challenge_set(cfg: Mapping[str, Any]) -> list[RAGSample]:
    """按配置生成挑战集（三类各 `n_base_per_type` 个基样本 → 污染版 + 对照版成对）。

    `cfg` 为 `configs/challenge.yaml` 与 `configs/default.yaml` 的合并结果（由 CLI 合并），需要
    `n_base_per_type`、`data_seed_base`、`operators`、`dataset` 四个键。构造种子 =
    `data_seed_base + 原始行索引`，与模型种子完全解耦（规范 §4）。每类不足目标数量时在日志与
    `construct_log.json` 中如实报告（指南 §6 的降级约定），不凑数、不硬塞。
    """
    from .reader import HOTPOTQA_CONFIG, HOTPOTQA_ID, load_dataset_split, normalize

    n_base = int(cfg.get("n_base_per_type", 50))
    seed_base = int(cfg.get("data_seed_base", DATA_SEED_BASE))
    operators = dict(cfg.get("operators") or {})
    hotpot_cfg = dict((cfg.get("dataset") or {}).get("hotpotqa") or {})
    splits = list(hotpot_cfg.get("splits") or ["validation"])

    view = load_dataset_split(str(hotpot_cfg.get("id", HOTPOTQA_ID)),
                              hotpot_cfg.get("config", HOTPOTQA_CONFIG),
                              split=str(splits[-1]), revision=hotpot_cfg.get("revision"))
    examples = [normalize(row, "hotpotqa", row_index=idx, hf_split=view.split)
                for idx, row in enumerate(view)]
    dataset_info = view.provenance()
    pool = build_pool(examples, start=len(examples) // 2, size=400)

    samples: list[RAGSample] = []
    cursor = 0
    for challenge_type in CHALLENGE_OPERATORS:
        params = dict(operators.get(challenge_type) or {})
        params["_pool"] = pool
        fn = _operator_fn(challenge_type)
        made = unsuitable = 0
        while made < n_base and cursor < len(examples):
            example = examples[cursor]
            cursor += 1
            seed = seed_base + int(example.get("row_index", 0))
            result = fn(example, params, seed)
            if result is None:
                unsuitable += 1
                continue
            samples.append(_to_sample(example, result, is_control=False,
                                      operator=challenge_type, dataset=dataset_info))
            samples.append(_to_sample(example, make_control(example, seed), is_control=True,
                                      operator=challenge_type, dataset=dataset_info))
            made += 1
        if made < n_base:
            LOG.error("挑战类型 %s 只构造出 %d/%d 个基样本（跳过 %d 个不适用样本）；"
                      "按指南 §6 降级并需在报告中如实说明", challenge_type, made, n_base, unsuitable)
        else:
            LOG.info("挑战类型 %s 构造完成：基样本 %d，样本 %d（污染版 + 对照版）",
                     challenge_type, made, made * 2)
    return samples
