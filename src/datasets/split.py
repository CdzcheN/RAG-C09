#!/usr/bin/env python3
"""抽样与划分：常规集抽样、按基样本切分 dev-fit / dev-test。

负责人: A
对应文档: docs/数据构造规范.md §1
状态: 已实现（T1.1/T3.1）

纪律: 抽样种子显式传入并写入产物元信息；划分以**基样本**为单位，污染版与对照版必须同侧
（否则测试集里出现同一基样本的另一版本，造成数据泄漏，规范 §6）。
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import RAGSample
from .reader import SQUAD_ID, HOTPOTQA_ID, HOTPOTQA_CONFIG, load_dataset_split, normalize, source_of

LOG = get_logger(__name__)


def base_key(sample: RAGSample) -> str:
    """基样本键：同一基样本的污染版与对照版具有相同的键。"""
    provenance = dict(sample.provenance or {})
    key = provenance.get("base_sample_key")
    if key:
        return str(key)
    return "-".join(sample.sample_id.split("-")[:-1])


def passages_of(example: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """规范化样本的候选段落 → 契约 `passages_for_retrieval` 的 `{title, text}` 形式。"""
    return tuple({"title": str(p.get("title", "")), "text": " ".join(str(s) for s in (p.get("sentences") or []))}
                 for p in (example.get("context") or []))


def sample_normal_set(cfg: Mapping[str, Any], seed: int) -> list[RAGSample]:
    """从 SQuAD v2 dev 与 HotpotQA validation 各抽样 N 条构成常规集。

    说明: 只抽**可回答**样本（SQuAD 不可回答问题无支撑证据句，不满足契约 §2.1 对 `gold_context`
    的要求）；抽样用固定种子，可复现。常规集与挑战集可能命中同一原始行，两者位于不同产物文件，
    文件内 `sample_id` 仍唯一（契约 §6 的主键规则以单文件为范围）。
    常规集的标签需模型答案产出后由 `challenge_builder.derive_label(..., is_control=True)` 回填，
    此处 `is_hallucination` 先记 0 并标 `label_source="pending_pipeline"`。
    """
    n_per = int((cfg.get("sampling") or {}).get("normal_per_dataset", 300))
    dataset_cfg = dict(cfg.get("dataset") or {})
    rng = random.Random(seed)
    samples: list[RAGSample] = []

    plan = (("squad", dataset_cfg.get("squad") or {}, SQUAD_ID, None),
            ("hotpotqa", dataset_cfg.get("hotpotqa") or {}, HOTPOTQA_ID, HOTPOTQA_CONFIG))
    for source, ds_conf, default_id, default_config in plan:
        splits = list(ds_conf.get("splits") or ["validation"])
        view = load_dataset_split(str(ds_conf.get("id", default_id)),
                                  ds_conf.get("config", default_config),
                                  split=str(splits[-1]), revision=ds_conf.get("revision"))
        examples = [normalize(row, source, row_index=idx, hf_split=view.split)
                    for idx, row in enumerate(view)]
        answerable = [ex for ex in examples
                      if not ex.get("is_impossible") and ex.get("gold_context") and ex.get("answer")]
        if len(answerable) < n_per:
            LOG.warning("%s 可回答样本只有 %d 条，少于目标 %d 条；按实际数量抽样",
                        source, len(answerable), n_per)
        picked = sorted(rng.sample(range(len(answerable)), min(n_per, len(answerable))))
        dataset_info = view.provenance()
        for idx in picked:
            example = answerable[idx]
            row_index = int(example.get("row_index", -1))
            split_tag = str(example.get("split_tag", "dev"))
            samples.append(RAGSample(
                sample_id=f"{source}-{split_tag}-{row_index:05d}-none",
                source=source,
                split="normal",
                challenge_type="none",
                question=str(example.get("question", "")),
                gold_answer=str(example.get("answer", "")),
                gold_context=tuple(str(s) for s in (example.get("gold_context") or [])),
                passages_for_retrieval=passages_of(example),
                is_hallucination=0,
                construct_params={"operator": "none", "seed": seed, "is_control": True,
                                  "label_source": "pending_pipeline",
                                  "label_rule": "由 derive_label(模型答案, is_control=True) 回填"},
                provenance={"row_index": row_index, "hf_split": example.get("hf_split", "dev"),
                            "source_id": example.get("id", ""), "dataset": dataset_info,
                            "base_sample_key": f"{source}-{split_tag}-{row_index:05d}"},
            ))
        LOG.info("常规集：%s 抽样 %d 条（候选 %d 条）", source, len(picked), len(answerable))
    return samples


def split_by_base_sample(samples: Sequence[RAGSample], ratio: Mapping[str, float],
                         seed: int) -> tuple[list[RAGSample], list[RAGSample]]:
    """按基样本切分，污染版与对照版必须同侧（避免数据泄漏）。

    `ratio` 见 `configs/default.yaml sampling.split_ratio`（默认 `dev_fit: 0.7`）。切分单位是
    基样本键，同键样本整体进入同一侧；返回顺序按 `sample_id` 排序，保证可复现。
    """
    fit_ratio = float(ratio.get("dev_fit", 0.7))
    groups: dict[str, list[RAGSample]] = {}
    for sample in samples:
        groups.setdefault(base_key(sample), []).append(sample)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_fit = int(round(len(keys) * fit_ratio))
    n_fit = max(0, min(len(keys), n_fit))
    fit_keys = set(keys[:n_fit])

    dev_fit: list[RAGSample] = []
    dev_test: list[RAGSample] = []
    for key in keys:
        target = dev_fit if key in fit_keys else dev_test
        target.extend(groups[key])
    dev_fit.sort(key=lambda s: s.sample_id)
    dev_test.sort(key=lambda s: s.sample_id)
    LOG.info("按基样本划分：基样本 %d 个（dev_fit=%d / dev_test=%d），样本 %d / %d",
             len(keys), len(fit_keys), len(keys) - len(fit_keys), len(dev_fit), len(dev_test))
    return dev_fit, dev_test
